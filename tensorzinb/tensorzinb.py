import warnings
import time

import os
import numpy as np
import scipy.sparse

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import tensorflow as tf

tf.get_logger().setLevel("ERROR")
tf.autograph.set_verbosity(0)

import tf_keras
from tf_keras.models import Model
from tf_keras.layers import Lambda, Input, Dense, RepeatVector, Reshape, Add
from tf_keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tf_keras.optimizers.legacy import RMSprop
from tf_keras import backend as K
from scipy.special import gammaln
import statsmodels.api as sm

# Reset Keras Session
def reset_keras():
    K.clear_session()


# --- Sparse helpers ---

def _scipy_to_tf_sparse(sp_matrix):
    """Convert scipy.sparse matrix to tf.SparseTensor."""
    coo = scipy.sparse.coo_matrix(sp_matrix)
    order = np.lexsort((coo.col, coo.row))
    indices = np.column_stack((coo.row[order], coo.col[order])).astype(np.int64)
    values = coo.data[order].astype(np.float32)
    return tf.SparseTensor(indices=indices, values=values, dense_shape=coo.shape)


def _to_tf_input(mat, is_sparse):
    """Convert dense or scipy sparse matrices to TF runtime inputs."""
    if is_sparse:
        return _scipy_to_tf_sparse(mat)
    return tf.convert_to_tensor(mat, dtype=tf.float32)


def _input_signature(shape, is_sparse):
    """Build a tf.data signature matching a model input."""
    if is_sparse:
        return tf.SparseTensorSpec(shape=shape, dtype=tf.float32)
    return tf.TensorSpec(shape=shape, dtype=tf.float32)


def _matrix_rank(mat):
    """Compute matrix rank for dense or sparse matrices."""
    if scipy.sparse.issparse(mat):
        k = min(mat.shape)
        if k <= 10000:
            ata = (mat.T @ mat).toarray()
            return np.linalg.matrix_rank(ata)
        else:
            return k
    return np.linalg.matrix_rank(mat)


def _sparse_std(mat, axis=0):
    """Compute std along axis for dense or sparse matrices."""
    if scipy.sparse.issparse(mat):
        mean_sq = np.asarray(mat.power(2).mean(axis=axis)).flatten()
        mean_ = np.asarray(mat.mean(axis=axis)).flatten()
        return np.sqrt(np.maximum(mean_sq - mean_ ** 2, 0))
    return np.std(mat, axis=axis)


def _sparse_col_mean(mat, col_idx):
    """Compute mean of a single column for dense or sparse matrices."""
    if scipy.sparse.issparse(mat):
        return mat[:, col_idx].toarray().mean()
    return np.mean(mat[:, col_idx])


def _sparse_dot(mat, vec):
    """Matrix-vector or matrix-matrix multiply, sparse-compatible."""
    if scipy.sparse.issparse(mat):
        return np.asarray(mat.dot(vec))
    return np.dot(mat, vec)


class SparseDense(tf_keras.layers.Layer):
    """Dense layer that supports tf.SparseTensor inputs via sparse_dense_matmul."""
    def __init__(self, units, use_bias=False, name=None, weights=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.units = units
        self._init_weights = weights

    def build(self, input_shape):
        init = (tf_keras.initializers.Constant(self._init_weights[0])
                if self._init_weights else 'glorot_uniform')
        self.kernel = self.add_weight(
            name='kernel',
            shape=(input_shape[-1], self.units),
            initializer=init,
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs):
        if isinstance(inputs, tf.SparseTensor):
            return tf.sparse.sparse_dense_matmul(inputs, self.kernel)
        return tf.matmul(inputs, self.kernel)


class ZINBLogLik:
    def __init__(
        self, pi=None, log_theta=None, nb_only=False, scope="zinb/", zero_threshold=1e-8
    ):
        self.pi = pi
        self.zero_threshold = zero_threshold
        self.scope = scope
        self.log_theta = log_theta
        self.nb_only = nb_only
        self.y = None
        self.llf = None

    def loss(self, y_true, y_pred):
        with tf.name_scope(self.scope):
            result, llf = self._loss_components(
                y_true,
                y_pred,
                self.pi,
                self.log_theta,
                self.nb_only,
                self.zero_threshold,
            )
            self.llf = llf
            self.y = tf.cast(y_true, tf.float32)
        return result

    @staticmethod
    def _elementwise_nll(y_true, y_pred, pi, log_theta, nb_only, zero_threshold):
        """Per-observation negative log-likelihood (shape [n, d]), no reduction."""
        y = tf.cast(y_true, tf.float32)
        mu = tf.cast(y_pred, tf.float32)
        log_theta = tf.cast(log_theta, tf.float32)
        theta = tf.math.exp(log_theta)

        t1 = tf.math.lgamma(y + theta)
        t2 = -tf.math.lgamma(theta)
        t3 = theta * log_theta
        t4 = y * mu
        ty = tf.reduce_logsumexp(tf.stack([log_theta, mu], axis=0), axis=0)
        t5 = -(theta + y) * ty

        if nb_only:
            return -(t1 + t2 + t3 + t4 + t5)

        pi = tf.cast(pi, tf.float32)
        log_q0 = -tf.nn.softplus(-pi)
        log_q1 = log_q0 - pi

        nb_case = -(t1 + t2 + t3 + t4 + t5 + log_q1)

        p1 = theta * (log_theta - ty) + log_q1
        zero_case = -tf.reduce_logsumexp(tf.stack([log_q0, p1], axis=0), axis=0)
        return tf.where(tf.less(y, zero_threshold), zero_case, nb_case)

    @staticmethod
    def _loss_components(y_true, y_pred, pi, log_theta, nb_only, zero_threshold,
                         sample_weight=None):
        result = ZINBLogLik._elementwise_nll(
            y_true, y_pred, pi, log_theta, nb_only, zero_threshold)
        if sample_weight is not None:
            w = tf.cast(tf.reshape(sample_weight, [-1, 1]), tf.float32)
            llf = tf.reduce_sum(result * w, axis=0) / tf.reduce_sum(w)
        else:
            llf = tf.reduce_mean(result, axis=0)
        return tf.reduce_sum(llf), llf


class PredictionCallback(tf_keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.weights = []

    def on_epoch_end(self, epoch, logs={}):
        ws = (
            np.concatenate([np.array(w.flatten()) for w in self.model.get_weights()])
        ).flatten()
        self.weights.append(ws)


class TensorZINBTrainingModel(Model):
    def __init__(self, inputs, outputs, nb_only=False, zero_threshold=1e-8,
                 sample_weight=None):
        super().__init__(inputs=inputs, outputs=outputs)
        self.nb_only = nb_only
        self.zero_threshold = zero_threshold
        self._sample_weight = (
            tf.constant(sample_weight, dtype=tf.float32)
            if sample_weight is not None else None
        )

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            outputs = self(x, training=True)
            if self.nb_only:
                pred_mu, pred_theta = outputs
                loss, _ = ZINBLogLik._loss_components(
                    y, pred_mu, None, pred_theta, True,
                    self.zero_threshold, self._sample_weight,
                )
            else:
                pred_mu, pred_pi, pred_theta = outputs
                loss, _ = ZINBLogLik._loss_components(
                    y, pred_mu, pred_pi, pred_theta, False,
                    self.zero_threshold, self._sample_weight,
                )

        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return {"loss": loss}


class ReduceLROnPlateauSkip(ReduceLROnPlateau):
    def __init__(
        self,
        monitor="val_loss",
        factor=0.1,
        patience=10,
        verbose=0,
        mode="auto",
        min_delta=1e-4,
        cooldown=0,
        min_lr=0,
        num_epoch_skip=3,
        **kwargs,
    ):
        self.num_epoch_skip = num_epoch_skip

        super(ReduceLROnPlateauSkip, self).__init__(
            monitor=monitor,
            factor=factor,
            patience=patience,
            verbose=verbose,
            mode=mode,
            min_delta=min_delta,
            cooldown=cooldown,
            min_lr=min_lr,
            **kwargs,
        )

    def on_epoch_end(self, epoch, logs=None):
        if epoch < self.num_epoch_skip:
            return
        super(ReduceLROnPlateauSkip, self).on_epoch_end(epoch, logs)


class TensorZINB:
    def __init__(
        self,
        endog,
        exog,
        exog_c=None,
        exog_infl=None,
        exog_infl_c=None,
        same_dispersion=False,
        nb_only=False,
        sample_weight=None,
        **kwargs,
    ):
        self.endog = endog
        if scipy.sparse.issparse(endog):
            self.endog = endog.toarray()
        if len(self.endog.shape) == 1:
            self.num_sample = len(self.endog)
            self.num_out = 1
            self.endog = self.endog.reshape((-1, 1))
        else:
            self.num_sample, self.num_out = np.shape(self.endog)

        if sample_weight is not None:
            self.sample_weight = np.asarray(sample_weight, dtype=np.float64).ravel()
            if self.sample_weight.shape[0] != self.num_sample:
                raise ValueError(
                    f"sample_weight length ({self.sample_weight.shape[0]}) "
                    f"must match num_sample ({self.num_sample}).")
            self.effective_n = float(np.sum(self.sample_weight))
        else:
            self.sample_weight = None
            self.effective_n = float(self.num_sample)

        df_model = 0
        df_model_c = 0

        self.exog = exog
        self._exog_is_sparse = scipy.sparse.issparse(exog)
        df_model = _matrix_rank(exog)
        if self._exog_is_sparse:
            self.k_exog = exog.shape[1]
        elif len(exog.shape) == 1:
            self.k_exog = 1
            self.exog = exog.reshape((-1, 1))
        else:
            self.k_exog = exog.shape[1]

        self.exog_c = exog_c
        if exog_c is None:
            self.k_exog_c = 0
            self._no_exog_c = True
            self._exog_c_is_sparse = False
            self.exog_c = np.zeros((self.num_sample, self.k_exog_c), dtype=np.float64)
        else:
            self._exog_c_is_sparse = scipy.sparse.issparse(exog_c)
            self.k_exog_c = exog_c.shape[1]
            self._no_exog_c = False
            if self.k_exog_c > 0:
                df_model_c = df_model_c + _matrix_rank(exog_c)

        self.nb_only = nb_only
        if exog_infl is None and exog_infl_c is None:
            self.nb_only = True

        self.exog_infl = exog_infl
        if exog_infl is None or self.nb_only:
            self.k_exog_infl = 0
            self._no_exog_infl = True
            self._exog_infl_is_sparse = False
            self.exog_infl = np.ones(
                (self.num_sample, self.k_exog_infl), dtype=np.float64
            )
        else:
            self._exog_infl_is_sparse = scipy.sparse.issparse(exog_infl)
            self.k_exog_infl = exog_infl.shape[1]
            self._no_exog_infl = False
            if self.k_exog_infl > 0:
                df_model = df_model + _matrix_rank(exog_infl)

        self.exog_infl_c = exog_infl_c
        if exog_infl_c is None or self.nb_only:
            self.k_exog_infl_c = 0
            self._no_exog_infl_c = True
            self._exog_infl_c_is_sparse = False
            self.exog_infl_c = np.ones(
                (self.num_sample, self.k_exog_infl_c), dtype=np.float64
            )
        else:
            self._exog_infl_c_is_sparse = scipy.sparse.issparse(exog_infl_c)
            self.k_exog_infl_c = exog_infl_c.shape[1]
            self._no_exog_infl_c = False
            if self.k_exog_infl_c > 0:
                df_model_c = df_model_c + _matrix_rank(exog_infl_c)

        df_model_each = df_model + df_model_c
        df_model = df_model * self.num_out + df_model_c

        self.same_dispersion = same_dispersion
        if same_dispersion:
            self.k_disperson = 1
            df_model = df_model + 1
        else:
            self.k_disperson = self.num_out
            df_model = df_model + self.num_out
        df_model_each = df_model_each + 1
        self.df_model = df_model
        self.df_model_each = df_model_each

        self.loglike_method = "nb2"

    def _make_runtime_inputs(self):
        return [
            _to_tf_input(self.exog, self._exog_is_sparse),
            _to_tf_input(self.exog_c, self._exog_c_is_sparse),
            _to_tf_input(self.exog_infl, self._exog_infl_is_sparse),
            _to_tf_input(self.exog_infl_c, self._exog_infl_c_is_sparse),
            tf.ones((self.num_sample, 1), dtype=tf.float32),
        ]

    def _make_training_dataset(self):
        inputs = self._make_runtime_inputs()
        output_signature = (
            (
                _input_signature((self.num_sample, self.k_exog), self._exog_is_sparse),
                _input_signature((self.num_sample, self.k_exog_c), self._exog_c_is_sparse),
                _input_signature((self.num_sample, self.k_exog_infl), self._exog_infl_is_sparse),
                _input_signature((self.num_sample, self.k_exog_infl_c), self._exog_infl_c_is_sparse),
                tf.TensorSpec(shape=(self.num_sample, 1), dtype=tf.float32),
            ),
            tf.TensorSpec(shape=self.endog.shape, dtype=tf.float32),
        )

        def gen():
            while True:
                yield tuple(inputs), self.endog.astype(np.float32)

        return tf.data.Dataset.from_generator(gen, output_signature=output_signature)

    def fit(
        self,
        init_weights={},
        init_method="poi",
        device_type="CPU",
        device_name=None,
        return_history=False,
        epochs=5000,
        learning_rate=0.01,
        num_epoch_skip=3,
        is_early_stop=True,
        is_reduce_lr=True,
        min_delta_early_stop=0.05,
        patience_early_stop=50,
        factor_reduce_lr=0.8,
        patience_reduce_lr=10,
        min_lr=0.001,
        reset_keras_session=False,
        **kwargs,
    ):
        if device_name is None:
            devices = tf.config.list_logical_devices(device_type)
            device_name = devices[0].name

        num_sample = self.num_sample
        num_out = self.num_out
        num_feat = self.k_exog
        num_feat_infl = self.k_exog_infl
        num_feat_c = self.k_exog_c
        num_feat_infl_c = self.k_exog_infl_c
        num_dispersion = self.k_disperson

        # initiate weights
        if len(init_weights) == 0:
            if init_method == "poi":
                init_weights = self._poisson_init()
            elif init_method == "nb":
                init_weights = self._nb_init(device_name=device_name)

        # use distinct names to retrieve weights from layers
        weight_keys = ["x_mu", "z_mu", "x_pi", "z_pi", "theta"]

        if reset_keras_session:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=DeprecationWarning)
                reset_keras()
                K.clear_session()

        any_sparse = (self._exog_is_sparse or self._exog_c_is_sparse or
                      self._exog_infl_is_sparse or self._exog_infl_c_is_sparse)

        with tf.device(device_name):
            inputs = Input(shape=(num_feat,), sparse=self._exog_is_sparse)
            inputs_infl = Input(shape=(num_feat_infl,), sparse=self._exog_infl_is_sparse)
            inputs_c = Input(shape=(num_feat_c,), sparse=self._exog_c_is_sparse)
            inputs_infl_c = Input(shape=(num_feat_infl_c,), sparse=self._exog_infl_c_is_sparse)
            inputs_theta = Input(shape=(1,))

            # select Dense or SparseDense based on input sparsity
            DenseCls_exog = SparseDense if self._exog_is_sparse else Dense
            DenseCls_exog_c = SparseDense if self._exog_c_is_sparse else Dense
            DenseCls_exog_infl = SparseDense if self._exog_infl_is_sparse else Dense
            DenseCls_exog_infl_c = SparseDense if self._exog_infl_c_is_sparse else Dense

            if 'x_mu' in init_weights:
                x = DenseCls_exog(num_out, use_bias=False, name='x_mu', weights=[init_weights['x_mu']])(inputs)
            else:
                x = DenseCls_exog(num_out, use_bias=False, name='x_mu')(inputs)
            if num_feat_c>0:
                if 'z_mu' in init_weights:
                    x_c = DenseCls_exog_c(1, use_bias=False, name='z_mu', weights=[init_weights['z_mu']])(inputs_c)
                else:
                    x_c = DenseCls_exog_c(1, use_bias=False, name='z_mu')(inputs_c)
                predictions = Add()([x,x_c])
            else:
                predictions = x

            if self.nb_only:
                pi = None
            else:
                if 'x_pi' in init_weights:
                    x_infl = DenseCls_exog_infl(num_out, use_bias=False, name='x_pi', weights=[init_weights['x_pi']])(inputs_infl)
                else:
                    x_infl = DenseCls_exog_infl(num_out, use_bias=False, name='x_pi')(inputs_infl)

                if num_feat_infl_c>0:
                    if 'z_pi' in init_weights:
                        x_infl_c = DenseCls_exog_infl_c(1, use_bias=False, name='z_pi', weights=[init_weights['z_pi']])(inputs_infl_c)
                    else:
                        x_infl_c = DenseCls_exog_infl_c(1, use_bias=False, name='z_pi')(inputs_infl_c)
                    pi = Add()([x_infl, x_infl_c])
                else:
                    pi = x_infl

            if 'theta' in init_weights:
                theta0 = Dense(num_dispersion, use_bias=False, name='theta', weights=[init_weights['theta']])(inputs_theta)
            else:
                theta0 = Dense(num_dispersion, use_bias=False, name='theta')(inputs_theta)
            if self.same_dispersion:
                theta = Reshape((num_out,))(RepeatVector(num_out)(theta0))
            else:
                theta = theta0

            model_inputs = [inputs, inputs_c, inputs_infl, inputs_infl_c, inputs_theta]
            model_outputs = [predictions, theta] if self.nb_only else [predictions, pi, theta]
            model = TensorZINBTrainingModel(
                inputs=model_inputs,
                outputs=model_outputs,
                nb_only=self.nb_only,
                zero_threshold=1e-8,
                sample_weight=self.sample_weight,
            )

            opt = RMSprop(learning_rate=learning_rate)
            model.compile(optimizer=opt, run_eagerly=True)

            callbacks = []
            if is_early_stop:
                early_stop = EarlyStopping(
                    monitor="loss",
                    min_delta=min_delta_early_stop / self.effective_n,
                    patience=patience_early_stop,
                    mode="min",
                )
                callbacks.append(early_stop)

            if is_reduce_lr:
                reduce_lr = ReduceLROnPlateauSkip(
                    monitor="loss",
                    factor=factor_reduce_lr,
                    patience=patience_reduce_lr,
                    min_lr=min_lr,
                    num_epoch_skip=num_epoch_skip,
                )
                callbacks.append(reduce_lr)

            if return_history:
                # this get all weights over epoch
                get_weights = PredictionCallback()
                callbacks.append(get_weights)

            runtime_inputs = self._make_runtime_inputs()
            tf_endog = tf.convert_to_tensor(self.endog.astype(np.float32), dtype=tf.float32)
            # TODO: Add optional minibatch training to reduce peak memory use on very large inputs.
            dataset = self._make_training_dataset() if any_sparse else None

            # TODO: FIX this. code randomly crashes on apple silicon M1/M2 with
            # error `Incompatible shapes`. code usually runs fine after second try.
            # similar to this issue https://developer.apple.com/forums/thread/701985
            losses = None
            cpu_time = None
            last_error = None
            for i in range(10):
                try:
                    start_time = time.time()
                    if any_sparse:
                        losses = model.fit(
                            dataset,
                            callbacks=callbacks,
                            steps_per_epoch=1,
                            epochs=epochs,
                            verbose=0,
                        )
                    else:
                        losses = model.fit(
                            [
                                self.exog,
                                self.exog_c,
                                self.exog_infl,
                                self.exog_infl_c,
                                np.ones((num_sample, 1)),
                            ],
                            [self.endog],
                            callbacks=callbacks,
                            batch_size=num_sample,
                            epochs=epochs,
                            verbose=0,
                        )
                    cpu_time = time.time() - start_time
                    break
                except Exception as e:
                    last_error = e
                    print(model.summary())
                    print(e)
                    continue

            if losses is None:
                raise RuntimeError("TensorZINB training failed after 10 attempts") from last_error

            tf_sw = (tf.constant(self.sample_weight, dtype=tf.float32)
                     if self.sample_weight is not None else None)

            eval_outputs = model(runtime_inputs, training=False)
            if self.nb_only:
                pred_mu, pred_theta = eval_outputs
                _, llft = ZINBLogLik._loss_components(
                    tf_endog, pred_mu, None, pred_theta, True,
                    model.zero_threshold, tf_sw,
                )
            else:
                pred_mu, pred_pi, pred_theta = eval_outputs
                _, llft = ZINBLogLik._loss_components(
                    tf_endog, pred_mu, pred_pi, pred_theta, False,
                    model.zero_threshold, tf_sw,
                )
            llft = llft.numpy()

            if self.sample_weight is not None:
                w = self.sample_weight.reshape(-1, 1)
                llfs = -(llft * self.effective_n
                         + np.sum(w * gammaln(self.endog + 1), axis=0))
            else:
                llfs = -(llft * num_sample + np.sum(gammaln(self.endog + 1), axis=0))
            aics = -2 * (llfs - self.df_model_each)

            llf = np.sum(llfs)
            aic = -2 * (llf - self.df_model)

            # get weights
            weights = model.get_weights()

            names_t = [
                (weight.name).split("/")[0]
                for layer in model.layers
                for weight in layer.weights
            ]
            # tf layer weight has subscript in names
            weight_names = []
            for name in names_t:
                matched = name
                for nt in weight_keys:
                    if nt in name:
                        matched = nt
                        break
                weight_names.append(matched)

            weights_dict = dict(zip(weight_names, weights))

            res = {
                "llf_total": llf,
                "llfs": llfs,
                "aic_total": aic,
                "aics": aics,
                "df_model_total": self.df_model,
                "df": self.df_model_each,
                "weights": weights_dict,
                "cpu_time": cpu_time,
                "num_sample": self.effective_n,
                "epochs": len(losses.history["loss"]),
            }

            if return_history:
                res["loss_history"] = losses.history["loss"]
                res["weights_history"] = np.array(get_weights.weights)

        return res

    # https://github.com/statsmodels/statsmodels/blob/main/statsmodels/discrete/discrete_model.py#L3691
    def _estimate_dispersion(self, mu, resid, df_resid=None, loglike_method="nb2",
                              sample_weight=None):
        if sample_weight is not None:
            w = sample_weight.ravel()
            sum_w = np.sum(w)
            if loglike_method == "nb2":
                a = np.sum(w * (resid**2 / mu - 1) / mu) / sum_w
            else:
                a = np.sum(w * (resid**2 / mu - 1)) / sum_w
        else:
            if df_resid is None:
                df_resid = resid.shape[0]
            if loglike_method == "nb2":
                a = ((resid**2 / mu - 1) / mu).sum() / df_resid
            else:
                a = (resid**2 / mu - 1).sum() / df_resid
        return a

    def _compute_pi_init(self, nz_prob, p_nonzero, infl_prob_max=0.99):
        ww = 1 - min(nz_prob / p_nonzero, infl_prob_max)
        return -np.log(1 / ww - 1)

    def _poisson_init_each(
        self,
        Y,
        estimate_infl=True,
        eps=1e-10,
        maxiter=100,
        theta_lb=0.05,
        intercept_var_th=1e-3,
        infl_prob_max=0.99,
    ):
        if self.sample_weight is not None:
            _avg = lambda x: np.average(x, weights=self.sample_weight)
        else:
            _avg = np.mean
        find_poi_sol = True
        if self._exog_is_sparse or self.sample_weight is not None:
            # statsmodels Poisson cannot handle sparse exog or sample weights
            find_poi_sol = False
        else:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                try:
                    poi_mod = sm.Poisson(Y, self.exog).fit(
                        maxiter=maxiter, disp=False, warn_convergence=False
                    )
                    if np.isnan(poi_mod.params).any():
                        find_poi_sol = False
                    else:
                        mu = poi_mod.predict()
                        a = self._estimate_dispersion(
                            mu, poi_mod.resid, df_resid=poi_mod.df_resid
                        )
                        theta = 1 / max(a, theta_lb)
                        x_mu = np.reshape(poi_mod.params, (self.k_exog, 1))
                except:
                    find_poi_sol = False

        if not find_poi_sol:
            vs = _sparse_std(self.exog, axis=0)
            # find intercept index
            min_idx = np.argmin(vs)
            if vs[min_idx] < intercept_var_th:
                x_mu = np.zeros((self.k_exog, 1))
                mu = _avg(Y)
                x_mu[min_idx] = np.log(mu) / _sparse_col_mean(self.exog, min_idx)
                resid = Y - mu
                a = self._estimate_dispersion(mu, resid,
                                              sample_weight=self.sample_weight)
                if np.isnan(a) or np.isinf(a):
                    a = theta_lb
                theta = 1 / max(a, theta_lb)
            else:
                x_mu = None
                return {}

        weights = {"x_mu": x_mu, "theta": np.array([np.log(theta)]).reshape((-1, 1))}

        if not self._no_exog_infl and estimate_infl:
            pred = np.maximum(mu, 10 * eps)
            p_nb_zero = np.power(theta / (theta + pred + eps), theta)
            p_nonzero = 1 - (_avg(p_nb_zero) if np.ndim(p_nb_zero) > 0
                             else float(p_nb_zero))

            # find intercept index
            vs = _sparse_std(self.exog_infl, axis=0)
            min_idx = np.argmin(vs)
            if vs[min_idx] < intercept_var_th:
                nz_prob = _avg(Y > 0)
                x_pi = np.zeros((self.k_exog_infl, 1))
                fv = _sparse_col_mean(self.exog_infl, min_idx)
                w_pi = self._compute_pi_init(
                    nz_prob, p_nonzero, infl_prob_max=infl_prob_max
                )
                x_pi[min_idx] = w_pi / fv
                weights["x_pi"] = x_pi

        return weights

    def _poisson_init(
        self,
        eps=1e-10,
        maxiter=100,
        theta_lb=0.05,
        intercept_var_th=1e-3,
        infl_prob_max=0.99,
    ):
        x_mu = []
        x_pi = []
        theta = []
        return_x_pi = True
        return_theta = True
        for i in range(self.num_out):
            w = self._poisson_init_each(
                self.endog[:, i],
                eps=eps,
                maxiter=maxiter,
                theta_lb=theta_lb,
                intercept_var_th=intercept_var_th,
                infl_prob_max=infl_prob_max,
            )

            if len(w) == 0:
                return {}

            if "x_mu" in w:
                x_mu.append(w["x_mu"])
            else:
                return {}

            if "x_pi" in w:
                x_pi.append(w["x_pi"])
            else:
                return_x_pi = False

            if "theta" in w:
                theta.append(w["theta"])
            else:
                return_theta = False
        weights = {"x_mu": np.concatenate(x_mu, axis=1)}
        if return_x_pi:
            weights["x_pi"] = np.concatenate(x_pi, axis=1)
        if return_theta:
            t = np.concatenate(theta, axis=1)
            if self.same_dispersion:
                weights["theta"] = np.array(np.mean(t)).reshape((-1, 1))
            else:
                weights["theta"] = t
        return weights

    def _nb_init(self, infl_prob_max=0.99, intercept_var_th=1e-3, device_name=None):
        nb_mod = TensorZINB(
            self.endog,
            self.exog,
            exog_c=self.exog_c,
            same_dispersion=self.same_dispersion,
            nb_only=True,
            sample_weight=self.sample_weight,
        )
        nb_res = nb_mod.fit(init_method="poi", device_name=device_name)
        weights = nb_res["weights"]

        if self._no_exog_infl:
            return weights
        # find intercept index
        vs = _sparse_std(self.exog_infl, axis=0)
        min_idx = np.argmin(vs)
        # do not compute logit weight if there is no intercept
        if vs[min_idx] < intercept_var_th:
            x_pi = np.zeros((self.k_exog_infl, self.num_out))
            fv = _sparse_col_mean(self.exog_infl, min_idx)
            if self.same_dispersion:
                theta = np.exp(
                    np.array(list(weights["theta"].flatten()) * self.num_out)
                )
            else:
                theta = np.exp(weights["theta"].flatten())

            mu_c = 0
            if self.k_exog_c > 0 and "z_mu" in weights:
                mu_c = _sparse_dot(self.exog_c, weights["z_mu"])

            if self.sample_weight is not None:
                _avg = lambda x: np.average(x, weights=self.sample_weight)
            else:
                _avg = np.mean

            for i in range(self.num_out):
                mu = _sparse_dot(self.exog, weights["x_mu"][:, i]) + mu_c
                mu = np.exp(mu)
                p_nonzero = 1 - _avg(np.power(theta[i] / (theta[i] + mu), theta[i]))
                nz_prob = _avg(self.endog[:, i] > 0)
                w_pi = self._compute_pi_init(
                    nz_prob, p_nonzero, infl_prob_max=infl_prob_max
                )
                x_pi[min_idx, i] = w_pi / fv

            weights["x_pi"] = x_pi

            if self.k_exog_infl_c > 0:
                weights["z_pi"] = np.zeros((self.k_exog_infl_c, 1))

        return weights
