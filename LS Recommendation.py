from os.path import join
import abc
import time
import os
import pickle as pkl
import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1.nn import dynamic_rnn
from tensorflow.keras.layers import Input, GRU, Lambda, Dense
from tensorflow.keras.layers import concatenate
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K
import random
import sklearn.cluster as skc
from collections import Counter
tf.disable_v2_behavior()



from sklearn.metrics import (
    roc_auc_score,
    log_loss,
    mean_squared_error,
    accuracy_score,
    f1_score,
)

def load_dict(filename):
    with open(filename, "rb") as f:
        f_pkl = pkl.load(f)
        return f_pkl


def mrr_score(y_true, y_score):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order)
    rr_score = y_true / (np.arange(len(y_true)) + 1)
    return np.sum(rr_score) / np.sum(y_true)

def ndcg_score(y_true, y_score, k=10):
    best = dcg_score(y_true, y_true, k)
    actual = dcg_score(y_true, y_score, k)
    return actual / best

def dcg_score(y_true, y_score, k=10):
    k = min(np.shape(y_true)[-1], k)
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    gains = 2 ** y_true - 1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)

def hit_score(y_true, y_score, k=10):
    ground_truth = np.where(y_true == 1)[0]
    argsort = np.argsort(y_score)[::-1][:k]
    for idx in argsort:
        if idx in ground_truth:
            return 1
    return 0

def cal_wauc(df, weight):

    weight["auc"] = df.groupby("users").apply(groupby_auc)
    wauc_score = (weight["weight"]*weight["auc"]).sum()
    weight.drop(columns="auc", inplace=True)

    return wauc_score

def groupby_auc(df):

    y_hat = df.preds
    y = df.labels
    return roc_auc_score(y, y_hat)

def cal_metric(labels, preds, metrics):
    res = {}
    if not metrics:
        return res
    for metric in metrics:
        if metric == "auc":
            auc = roc_auc_score(np.asarray(labels), np.asarray(preds))
            res["auc"] = round(auc, 4)
        elif metric == "rmse":
            rmse = mean_squared_error(np.asarray(labels), np.asarray(preds))
            res["rmse"] = np.sqrt(round(rmse, 4))
        elif metric == "logloss":
            # avoid logloss nan
            preds = [max(min(p, 1.0 - 10e-12), 10e-12) for p in preds]
            logloss = log_loss(np.asarray(labels), np.asarray(preds))
            res["logloss"] = round(logloss, 4)
        elif metric == "acc":
            pred = np.asarray(preds)
            pred[pred >= 0.5] = 1
            pred[pred < 0.5] = 0
            acc = accuracy_score(np.asarray(labels), pred)
            res["acc"] = round(acc, 4)
        elif metric == "f1":
            pred = np.asarray(preds)
            pred[pred >= 0.5] = 1
            pred[pred < 0.5] = 0
            f1 = f1_score(np.asarray(labels), pred)
            res["f1"] = round(f1, 4)
        elif metric == "mean_mrr":
            mean_mrr = np.mean(
                [
                    mrr_score(each_labels, each_preds)
                    for each_labels, each_preds in zip(labels, preds)
                ]
            )
            res["mean_mrr"] = round(mean_mrr, 4)
        elif metric.startswith("ndcg"):  # format like:  ndcg@2;4;6;8
            ndcg_list = [1, 2]
            ks = metric.split("@")
            if len(ks) > 1:
                ndcg_list = [int(token) for token in ks[1].split(";")]
            for k in ndcg_list:
                ndcg_temp = np.mean(
                    [
                        ndcg_score(each_labels, each_preds, k)
                        for each_labels, each_preds in zip(labels, preds)
                    ]
                )
                res["ndcg@{0}".format(k)] = round(ndcg_temp, 4)
        elif metric.startswith("hit"):  # format like:  hit@2;4;6;8
            hit_list = [1, 2]
            ks = metric.split("@")
            if len(ks) > 1:
                hit_list = [int(token) for token in ks[1].split(";")]
            for k in hit_list:
                hit_temp = np.mean(
                    [
                        hit_score(each_labels, each_preds, k)
                        for each_labels, each_preds in zip(labels, preds)
                    ]
                )
                res["hit@{0}".format(k)] = round(hit_temp, 4)
        elif metric == "group_auc":
            group_auc = np.mean(
                [
                    roc_auc_score(each_labels, each_preds)
                    for each_labels, each_preds in zip(labels, preds)
                ]
            )
            res["group_auc"] = round(group_auc, 4)
        else:
            raise ValueError("not define this metric {0}".format(metric))
    return res


def cal_weighted_metric(users, preds, labels, metrics):
    res = {}
    if not metrics:
        return res
    df = pd.DataFrame({'users': users, 'preds': preds, 'labels': labels})
    weight = df[["users", "labels"]].groupby("users").count().reset_index().set_index("users", drop=True).rename(
        columns={"labels": "weight"})
    weight["weight"] = weight["weight"] / weight["weight"].sum()
    for metric in metrics:
        if metric == 'wauc':
            wauc = cal_wauc(df, weight)
            res["wauc"] = round(wauc, 4)
        elif metric == 'wmrr':
            wmrr = cal_wmrr(df, weight)
            res["wmrr"] = round(wmrr, 4)
        elif metric.startswith("whit"):  # format like: whit@2;4;6;8
            hit_list = [1, 2]
            ks = metric.split("@")
            if len(ks) > 1:
                hit_list = [int(token) for token in ks[1].split(";")]
            whit_res = cal_whit(df, weight, hit_list)
            res.update(whit_res)
        elif metric.startswith("wndcg"):  # format like: wndcg@2;4;6;8
            ndcg_list = [1, 2]
            ks = metric.split("@")
            if len(ks) > 1:
                ndcg_list = [int(token) for token in ks[1].split(";")]
            wndcg_res = cal_wndcg(df, weight, ndcg_list)
            res.update(wndcg_res)
        else:
            raise ValueError("not define this metric {0}".format(metric))
    return res

def slice_2(x, index):
    return x[:, index, :]


def user_cjpa(hist_recent, user_long, att_size):
    neigb_num, size = K.int_shape(user_long)[1], K.int_shape(user_long)[2]
    metapath = slice_2(user_long, 0)
    metapath2 = slice_2(hist_recent, 0)
    inputs = tf.concat([metapath2, metapath], axis=-1)
    tf.convert_to_tensor(inputs)

    dense_layer_1 = Dense(att_size,
                          activation='relu',
                          kernel_initializer='glorot_normal',
                          kernel_regularizer=l2(0.001),
                          name='user_cjpa_layer_1')
    dense_layer_2 = Dense(1,
                          activation='relu',
                          kernel_initializer='glorot_normal',
                          kernel_regularizer=l2(0.001),
                          name='user_cjpa_layer_2')
    output = (dense_layer_1(inputs))

    output = (dense_layer_2(output))

    metapath = Lambda(slice_2, output_shape=(size,), arguments={'index': 0})(user_long)
    metapath2 = Lambda(slice_2, output_shape=(size,), arguments={'index': 0})(hist_recent)
    # metapath = Reshape((1, size))(metapath)
    inputs = concatenate([metapath2, metapath])

    dense_layer_1 = Dense(att_size,
                          activation='relu',
                          kernel_initializer='glorot_normal',
                          kernel_regularizer=l2(0.001),
                          name='user_cjpa_layer_1')
    dense_layer_2 = Dense(1,
                          activation='relu',
                          kernel_initializer='glorot_normal',
                          kernel_regularizer=l2(0.001),
                          name='user_cjpa_layer_2')
    output = (dense_layer_1(inputs))
    output = (dense_layer_2(output))

    for i in range(1, neigb_num):
        metapath = Lambda(slice_2, output_shape=(size,), arguments={'index': i})(user_long)
        metapath2 = Lambda(slice_2, output_shape=(size,), arguments={'index': i})(hist_recent)
        inputs = concatenate([metapath2, metapath])
        tmp_output = (dense_layer_1(inputs))
        tmp_output = (dense_layer_2(tmp_output))
        output = concatenate([output, tmp_output])

    atten = Lambda(lambda x: K.softmax(x), name='user_cjpa_softmax')(output)
    output = Lambda(lambda x: K.sum(x[0] * K.expand_dims(x[1], -1), 1))([hist_recent, atten])
    return output


class BaseModel:
    def __init__(self, iterator_creator, seed, graph=None):
        self.seed = seed
        tf.set_random_seed(seed)
        np.random.seed(seed)

        self.graph = graph if graph is not None else tf.Graph()
        self.iterator = iterator_creator(self.graph)
        self.train_num_ngs = 4
        self.dropout = [0.0, 0.0]
        self.layer_sizes = [100, 64]
        self.max_grad_norm = 0.5
        self.metrics = ['auc', 'logloss']
        self.pairwise_metrics = ['mean_mrr', 'ndcg@2;4;6', 'hit@2;4;6', "group_auc"]
        self.weighted_metrics = ['wauc']
        self.EARLY_STOP = 10
        self.user_vocab_length = len(load_dict(user_vocab))
        self.item_vocab_length = len(load_dict(item_vocab))
        self.cate_vocab_length = len(load_dict(cate_vocab))
        self.user_embedding_dim = user_embedding_dim
        self.item_embedding_dim = item_embedding_dim
        self.cate_embedding_dim = cate_embedding_dim
        self.max_seq_length = max_seq_length

        with self.graph.as_default():
            self.labels = tf.placeholder(tf.float32, [None, 1], name="label")
            self.users = tf.placeholder(tf.int32, [None], name="users")
            self.items = tf.placeholder(tf.int32, [None], name="items")
            self.cates = tf.placeholder(tf.int32, [None], name="cates")
            self.item_history = tf.placeholder(
                tf.int32, [None, self.max_seq_length], name="item_history"
            )
            self.item_cate_history = tf.placeholder(
                tf.int32, [None, self.max_seq_length], name="item_cate_history"
            )
            self.mask = tf.placeholder(
                tf.int32, [None, self.max_seq_length], name="mask"
            )
            self.time = tf.placeholder(tf.float32, [None], name="time")
            self.time_diff = tf.placeholder(
                tf.float32, [None, self.max_seq_length], name="time_diff"
            )
            self.time_from_first_action = tf.placeholder(
                tf.float32, [None, self.max_seq_length], name="time_from_first_action"
            )
            self.time_to_now = tf.placeholder(
                tf.float32, [None, self.max_seq_length], name="time_to_now"
            )
            self.attn_labels = tf.placeholder(tf.float32, [None, 1], name="attn_label")

            with tf.variable_scope("embedding", initializer=tf.truncated_normal_initializer(stddev=0.01, seed=None)):
                self.user_long_lookup = tf.get_variable(
                    name="user_long_embedding",
                    shape=[self.user_vocab_length, self.user_embedding_dim],
                    dtype=tf.float32,
                )
                self.user_short_lookup = tf.get_variable(
                    name="user_short_embedding",
                    shape=[self.user_vocab_length, self.user_embedding_dim],
                    dtype=tf.float32,
                )

                self.item_lookup = tf.get_variable(
                    name="item_embedding",
                    shape=[self.item_vocab_length, self.item_embedding_dim],
                    dtype=tf.float32,
                )
                self.cate_lookup = tf.get_variable(
                    name="cate_embedding",
                    shape=[self.cate_vocab_length, self.cate_embedding_dim],
                    dtype=tf.float32,
                )

            self.layer_params = []
            self.embed_params = []
            self.cross_params = []
            self.layer_keeps = tf.placeholder(tf.float32, name="layer_keeps")
            self.embedding_keeps = tf.placeholder(tf.float32, name="embedding_keeps")
            self.keep_prob_train = None
            self.keep_prob_test = None
            self.embedding_keep_prob_train = None
            self.embedding_keep_prob_test = None
            self.is_train_stage = tf.placeholder(tf.bool, shape=(), name="is_training")
            self.logit = self._build_graph()
            self.pred = self._get_pred(self.logit)
            self.loss = self._get_loss()
            self.saver = tf.train.Saver(max_to_keep=5)
            self.update = self._build_train_opt()
            self.extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            self.init_op = tf.global_variables_initializer()

        gpu_options = tf.GPUOptions(allow_growth=True)
        self.sess = tf.Session(graph = self.graph, config=tf.ConfigProto(gpu_options=gpu_options))
        self.sess.run(self.init_op)

    def _dropout(self, logit, keep_prob):
        return tf.nn.dropout(x=logit, keep_prob=keep_prob)

    def _get_pred(self, logit):
        pred = tf.sigmoid(logit)
        return pred

    def _add_norm(self):
        all_variables, embed_variables = (
            tf.trainable_variables(),
            tf.trainable_variables(self.sequential_scope._name + "/embedding"),
        )
        layer_params = list(set(all_variables) - set(embed_variables))
        self.layer_params.extend(layer_params)

    def _build_graph(self):
        dropout = [0.0, 0.0]
        self.keep_prob_train = 1 - np.array(self.dropout)
        self.keep_prob_test = np.ones_like(self.dropout)

        self.embedding_keep_prob_train = 1.0 - 0.0
        self.embedding_keep_prob_test = 1.0

        with tf.variable_scope("sequential") as self.sequential_scope:
            #self._build_embedding()
            self._lookup_from_embedding()
            model_output = self._build_seq_graph()
            logit = self.new_fcn_net(model_output, self.layer_sizes, scope="logit_fcn")
            self._add_norm()
            return logit


    def gen_feed_dict(self, data_dict):
        if not data_dict:
            return dict()
        feed_dict = {
            self.labels: data_dict["labels"],
            self.attn_labels: data_dict["attn_labels"],
            self.users: data_dict["users"],
            self.items: data_dict["items"],
            self.cates: data_dict["cates"],
            self.item_history: data_dict["item_history"],
            self.item_cate_history: data_dict["item_cate_history"],
            self.mask: data_dict["mask"],
            self.time: data_dict["time"],
            self.time_diff: data_dict["time_diff"],
            self.time_from_first_action: data_dict["time_from_first_action"],
            self.time_to_now: data_dict["time_to_now"],
        }
        return feed_dict

    def fit(self, train_file, valid_file, train_num_ngs, valid_num_ngs, eval_metric="group_auc"):
        train_sess = self.sess
        eval_info = list()
        best_metric, self.best_epoch = 0, 0
        for epoch in range(50):
            file_iterator = self.iterator.load_data_from_file(train_file, train_num_ngs, min_seq_length=1)
            epoch_loss = self.batch_train(file_iterator, train_sess)
            print("epoch {0:d} , total_loss: {1:.4f}".format(epoch, epoch_loss))
            valid_res = self.run_weighted_eval(valid_file, valid_num_ngs, fla=True)
            print("eval valid at epoch {0}: {1}".format(epoch, ",".join(
                ["" + str(key) + ":" + str(value) for key, value in valid_res.items()]), ))
            eval_info.append((epoch, valid_res))

            progress = False
            early_stop = self.EARLY_STOP
            if valid_res[eval_metric] > best_metric:
                best_metric = valid_res[eval_metric]
                self.best_epoch = epoch
                progress = True
            else:
                if early_stop > 0 and epoch - self.best_epoch >= early_stop:
                    print("early stop at epoch {0}!".format(epoch))
                    break
            if not os.path.exists(model_path):
                os.makedirs(model_path)
            if progress:
                checkpoint_path = self.saver.save(
                    sess=train_sess,
                    save_path=model_path + "epoch_" + str(epoch),
                )
        print(eval_info)
        print("best epoch: {0}".format(self.best_epoch))
        return self

    def train(self, sess, feed_dict):
        feed_dict[self.layer_keeps] = self.keep_prob_train
        feed_dict[self.embedding_keeps] = self.embedding_keep_prob_train
        feed_dict[self.is_train_stage] = True
        return sess.run(
            [
                self.update,
                self.extra_update_ops,
                self.loss,
                self.data_loss,
                self.regular_loss,
                self.contrastive_loss],
            feed_dict=feed_dict,
        )


    def batch_train(self, file_iterator, train_sess):
        step = 0
        epoch_loss = 0
        epoch_data_loss = 0
        epoch_regular_loss = 0
        epoch_contrastive_loss = 0
        for batch_data_input in file_iterator:
            batch_data_input = self.gen_feed_dict(batch_data_input)
            if batch_data_input:
                step_result = self.train(train_sess, batch_data_input)
                (_, _, step_loss, step_data_loss, step_regular_loss, step_contrastive_loss) = step_result
                epoch_loss += step_loss
                epoch_data_loss += step_data_loss
                epoch_regular_loss += step_regular_loss
                epoch_contrastive_loss += step_contrastive_loss
                step += 1
                if step % 10 == 0:
                    print(
                        "step {0:d} , total_loss: {1:.4f}, data_loss: {2:.4f}".format(
                            step, step_loss, step_data_loss
                        )
                    )
        return epoch_loss


    def run_weighted_eval(self, filename, num_ngs, fla, calc_mean_alpha=False, manual_alpha=False):
        load_sess = self.sess
        users = []
        preds = []
        labels = []
        group_preds = []
        group_labels = []
        if fla:
            group = 5
        else:
            group = 100
        if calc_mean_alpha:
            alphas = []

        for batch_data_input in self.iterator.load_data_from_file(
                filename,  num_ngs, min_seq_length=1
        ):
            batch_data_input = self.gen_feed_dict(batch_data_input)
            if batch_data_input:
                if not calc_mean_alpha:
                    step_user, step_pred, step_labels = self.eval_with_user(load_sess, batch_data_input)
                else:
                    step_user, step_pred, step_labels, step_alpha = self.eval_with_user_and_alpha(load_sess,
                                                                                                  batch_data_input)
                    alphas.extend(np.reshape(step_alpha, -1))
                users.extend(np.reshape(step_user, -1))
                preds.extend(np.reshape(step_pred, -1))
                labels.extend(np.reshape(step_labels, -1))
                group_preds.extend(np.reshape(step_pred, (-1, group)))
                group_labels.extend(np.reshape(step_labels, (-1, group)))

        res = cal_metric(labels, preds, self.metrics)
        res_pairwise = cal_metric(
            group_labels, group_preds, self.pairwise_metrics
        )
        res.update(res_pairwise)
        res_weighted = cal_weighted_metric(users, preds, labels, self.weighted_metrics)
        res.update(res_weighted)
        if calc_mean_alpha:
            if manual_alpha:
                alphas = alphas[0]
            res_alpha = cal_mean_alpha_metric(alphas, labels)
            res.update(res_alpha)
        return res

    def eval_with_user(self, sess, feed_dict):
        feed_dict[self.layer_keeps] = self.keep_prob_test
        feed_dict[self.embedding_keeps] = self.embedding_keep_prob_test
        feed_dict[self.is_train_stage] = False
        return sess.run([self.users,self.pred, self.labels],
                        feed_dict=feed_dict)


    def _get_loss(self):
        self.data_loss = self._compute_data_loss()
        self.regular_loss = self._compute_regular_loss()
        self.contrastive_loss = self._compute_contrastive_loss()
        #self.discrepancy_loss = self._compute_discrepancy_loss()

        self.loss = self.data_loss + self.contrastive_loss + self.regular_loss
        return self.loss

    def _compute_data_loss(self):
        group = self.train_num_ngs + 1
        logits = tf.reshape(self.logit, (-1, group))
        labels = tf.reshape(self.labels, (-1, group))
        softmax_pred = tf.nn.softmax(logits, axis=-1)
        boolean_mask = tf.equal(labels, tf.ones_like(labels))
        mask_paddings = tf.ones_like(softmax_pred)
        pos_softmax = tf.where(boolean_mask, softmax_pred, mask_paddings)
        data_loss = -group * tf.reduce_mean(tf.math.log(pos_softmax))
        return data_loss

    def _compute_long_contrastive_loss(self):
        distance = tf.reduce_sum(tf.square(tf.reshape(self.new_att_fea_long, [-1]) - tf.reshape(self.contra_hist_input, [-1])))
        contrastive_loss = tf.sigmoid(distance)
        contrastive_loss = tf.multiply(0.4, contrastive_loss)
        return contrastive_loss

    def _compute_short_contrastive_loss(self):
        distance = tf.reduce_sum(tf.square(tf.reshape(self.new_att_fea_long, [-1]) - tf.reshape(self.att_fea_short, [-1])))
        contrastive_loss = tf.sigmoid(distance)
        contrastive_loss = -tf.multiply(0.4, contrastive_loss)
        return contrastive_loss

    def _compute_contrastive_loss(self):
        contrastive_loss = self._compute_long_contrastive_loss() + self._compute_short_contrastive_loss()
        return contrastive_loss

    def _l2_loss(self):
        l2_loss = tf.zeros([1], dtype=tf.float32)
        # embedding_layer l2 loss
        for param in self.embed_params:
            l2_loss = tf.add(
                l2_loss, tf.multiply(0.0001, tf.nn.l2_loss(param))
            )
        params = self.layer_params
        for param in params:
            l2_loss = tf.add(
                l2_loss, tf.multiply(0.0001, tf.nn.l2_loss(param))
            )
        return l2_loss

    def _l1_loss(self):
        l1_loss = tf.zeros([1], dtype=tf.float32)
        # embedding_layer l2 loss
        for param in self.embed_params:
            l1_loss = tf.add(
                l1_loss, tf.multiply(0.0000, tf.norm(param, ord=1))
            )
        params = self.layer_params
        for param in params:
            l1_loss = tf.add(
                l1_loss, tf.multiply(0.0000, tf.norm(param, ord=1))
            )
        return l1_loss

    def _cross_l_loss(self):
        cross_l_loss = tf.zeros([1], dtype=tf.float32)
        for param in self.cross_params:
            cross_l_loss = tf.add(
                cross_l_loss, tf.multiply(0.000, tf.norm(param, ord=1))
            )
            cross_l_loss = tf.add(
                cross_l_loss, tf.multiply(0.0000, tf.norm(param, ord=2))
            )
        return cross_l_loss

    def _compute_regular_loss(self):
        regular_loss = self._l2_loss() + self._l1_loss() + self._cross_l_loss()
        return tf.reduce_sum(regular_loss)


    def _build_train_opt(self):
        train_step = self._train_opt()
        gradients, variables = zip(*train_step.compute_gradients(self._get_loss()))
        gradients = [None if gradient is None
                     else tf.clip_by_norm(gradient, self.max_grad_norm)
                     for gradient in gradients
                     ]
        return train_step.apply_gradients(zip(gradients, variables))

    def _train_opt(self):
        lr = 0.001
        train_step = tf.train.AdamOptimizer(lr)
        return train_step


    def _lookup_from_embedding(self):
        self.item_embedding = tf.nn.embedding_lookup(
            self.item_lookup, self.items
        )
        self.item_history_embedding = tf.nn.embedding_lookup(
            self.item_lookup, self.item_history
        )

        self.cate_embedding = tf.nn.embedding_lookup(
            self.cate_lookup, self.cates
        )
        self.cate_history_embedding = tf.nn.embedding_lookup(
            self.cate_lookup, self.item_cate_history
        )

        self.user_long_embedding = tf.nn.embedding_lookup(
            self.user_long_lookup, self.users
        )

        self.user_short_embedding = tf.nn.embedding_lookup(
            self.user_short_lookup, self.users
        )

        involved_users = tf.reshape(self.users, [-1])
        self.involved_users, _ = tf.unique(involved_users)
        self.involved_user_long_embedding = tf.nn.embedding_lookup(
            self.user_long_lookup, self.involved_users
        )
        self.embed_params.append(self.involved_user_long_embedding)
        self.involved_user_short_embedding = tf.nn.embedding_lookup(
            self.user_short_lookup, self.involved_users
        )
        self.embed_params.append(self.involved_user_short_embedding)

        involved_items = tf.concat(
            [
                tf.reshape(self.item_history, [-1]),
                tf.reshape(self.items, [-1]),
            ],
            -1,
        )
        self.involved_items, _ = tf.unique(involved_items)
        involved_item_embedding = tf.nn.embedding_lookup(
            self.item_lookup, self.involved_items
        )
        self.embed_params.append(involved_item_embedding)

        involved_cates = tf.concat(
            [
                tf.reshape(self.item_cate_history, [-1]),
                tf.reshape(self.cates, [-1]),
            ],
            -1,
        )
        self.involved_cates, _ = tf.unique(involved_cates)
        involved_cate_embedding = tf.nn.embedding_lookup(
            self.cate_lookup, self.involved_cates
        )
        self.embed_params.append(involved_cate_embedding)
        self.target_item_embedding = tf.concat(
            [self.item_embedding, self.cate_embedding], -1
        )

        self.user_long_embedding = self._dropout(
            self.user_long_embedding, keep_prob=self.embedding_keeps)
        self.user_short_embedding = self._dropout(
            self.user_short_embedding, keep_prob=self.embedding_keeps)
        self.item_history_embedding = self._dropout(
            self.item_history_embedding, keep_prob=self.embedding_keeps)
        self.cate_history_embedding = self._dropout(
            self.cate_history_embedding, keep_prob=self.embedding_keeps)
        self.target_item_embedding = self._dropout(
            self.target_item_embedding, keep_prob=self.embedding_keeps
        )

    def news_attention(self, user_emb, news_t, att_size):
        # latent_size = news_emb.shape[1].value
        neigb_num, size = K.int_shape(news_t)[1], K.int_shape(news_t)[2]
        metapath = Lambda(slice_2, output_shape=(size,), arguments={'index': 0})(news_t)
        # metapath = Reshape((1, size))(metapath)
        inputs = concatenate([user_emb, metapath])

        dense_layer_1 = Dense(att_size,
                              activation='relu',
                              kernel_initializer='glorot_normal',
                              kernel_regularizer=l2(0.001),
                              name='news_attention_layer_1')
        dense_layer_2 = Dense(1,
                              activation='relu',
                              kernel_initializer='glorot_normal',
                              kernel_regularizer=l2(0.001),
                              name='news_attention_layer_2')
        output = (dense_layer_1(inputs))
        output = (dense_layer_2(output))

        for i in range(1, neigb_num):
            metapath = Lambda(slice_2, output_shape=(size,), arguments={'index': i})(news_t)
            # metapath = Reshape((1, size))(metapath)
            inputs = concatenate([user_emb, metapath])
            tmp_output = (dense_layer_1(inputs))
            tmp_output = (dense_layer_2(tmp_output))
            output = concatenate([output, tmp_output])

        atten = Lambda(lambda x: K.softmax(x), name='news_attention_softmax')(output)
        output = Lambda(lambda x: K.sum(x[0] * K.expand_dims(x[1], -1), 1))([news_t, atten])
        return output

    def _build_seq_graph(self):
        with tf.variable_scope("clsr"):
            hist_input = tf.concat(
                [self.item_history_embedding, self.cate_history_embedding], 2
            ) #(?, 50, 40)

            #contra_hist = []
            #for i in range(batch_size):
                #db = skc.DBSCAN().fit(tf.Session().run(hist_input[1, :, :]))
                #count = Counter(db.labels_)
                #counts = sorted(count.items(), key=lambda x: x[1], reverse=True)
                #label = counts[1][0] if counts[0][0] == -1 else counts[0][0]
                #hist = list(hist_input[i, :, :][db.labels_ == label].flatten())
                #hist = tf.convert_to_tensor(np.array(hist).reshape((50, 40)))
                #contra_hist.apend(tf.expand_dims(hist, 0))
            #self.contra_hist_input = tf.reduce_sum(tf.concat(contra_hist, axis=0), 1) #(?, 40)
            self.ava_hist_input = tf.reduce_sum(hist_input, 1)  # (?, 40)
            #self.ava_hist_input_new = tf.tile(tf.expand_dims(self.ava_hist_input, 1), [1, 50, 1])

            for i in range(13):
                self.ava_hist_input_new = tf.tile(tf.expand_dims(self.ava_hist_input, 1), [1, 50, 1])
                self.hist_sim = self.ava_hist_input_new * hist_input #(?, 50, 40)
                self.hist_sim = tf.reduce_sum(self.hist_sim, -1) #(?, 50)
                sorted_hist_sim, indices = tf.nn.top_k(tf.sigmoid(self.hist_sim) ,k=5) #(?, 5)
                self.weight_hist = tf.gather(hist_input, indices, axis=1)[:, 0, :, :] * tf.tile(tf.expand_dims(sorted_hist_sim, -1), [1, 1, 40]) #(?, 5, 40)
                self.ava_hist_input += tf.reduce_sum(self.weight_hist, 1)
            self.contra_hist_input = self.ava_hist_input

            self.mask = self.mask
            self.real_mask = tf.cast(self.mask, tf.float32)
            self.sequence_length = tf.reduce_sum(self.mask, 1)

            with tf.variable_scope("long_term"):
                self.att_fea_long = self._attention_fcn(self.user_long_embedding, hist_input, max_iter) #(?, 50, 40)
                self.new_att_fea_long = tf.reduce_sum(self.att_fea_long, 1)

            with tf.variable_scope("short_term"):
                self.position = tf.math.cumsum(self.real_mask, axis=1, reverse=True)
                self.recent_mask = tf.logical_and(self.position >= 1, self.position <= 3)
                self.real_recent_mask = tf.where(self.recent_mask, tf.ones_like(self.recent_mask, dtype=tf.float32),
                                                 tf.zeros_like(self.recent_mask, dtype=tf.float32))

                self.hist_recent = tf.reduce_sum(hist_input * tf.expand_dims(self.real_recent_mask, -1),
                                                 1) / tf.reduce_sum(self.real_recent_mask, 1, keepdims=True) #(?, 40)

                self.hist_recent = tf.tile(tf.expand_dims(self.hist_recent, 1), [1, self.att_fea_long.shape[1], 1]) #(?, 50, 40)

                rnn_outputs, _ = dynamic_rnn(
                    tf.nn.rnn_cell.GRUCell(40),
                    inputs=hist_input,
                    sequence_length=self.sequence_length,
                    dtype=tf.float32,
                    scope="simple_gru",
                ) #(?, 50, 40)
                #rnn_outputs为key

                att_outputs_short = self.short_attention_fcn(self.att_fea_long, self.user_long_embedding, hist_input, rnn_outputs, att_size, max_iter) #(?, 50, 40)
                self.att_fea_short = tf.reduce_sum(att_outputs_short, 1) #(?, 40)

            with tf.name_scope("alpha"):
                if not False:
                    if True:
                        with tf.variable_scope("causal2"):
                            _, final_state = dynamic_rnn(
                                tf.nn.rnn_cell.GRUCell(40),
                                inputs=hist_input,
                                sequence_length=self.sequence_length,
                                dtype=tf.float32,
                                scope="causal2",
                            )

                        concat_all = tf.concat(
                            [
                                final_state,  # (?, 40)
                                self.target_item_embedding,  # (?, 40)
                                self.new_att_fea_long,
                                self.att_fea_short,
                                tf.expand_dims(self.time_to_now[:, -1], -1),
                            ],
                            1,
                        )
                    last_hidden_nn_layer = concat_all
                    alpha_logit = self.new_fcn_net(
                        last_hidden_nn_layer, att_fcn_layer_sizes, scope="fcn_alpha"
                    )
                    self.alpha_output = tf.sigmoid(alpha_logit)
                    user_embed = self.new_att_fea_long * self.alpha_output + self.att_fea_short * (
                                1.0 - self.alpha_output)

                    self.alpha_output_mean = self.alpha_output
                    error_with_category = self.alpha_output_mean - self.attn_labels

                    squared_error_with_category = tf.math.sqrt(
                        tf.math.squared_difference(tf.reshape(self.alpha_output_mean, [-1]),
                                                   tf.reshape(self.attn_labels, [-1])))

                model_output = tf.concat([user_embed, self.target_item_embedding], 1)
            return model_output

    def _active_layer(self, logit, activation, layer_idx=-1):
        if layer_idx >= 0 and False:
            logit = self._dropout(logit, self.layer_keeps[layer_idx])
        return self._activate(logit, activation, layer_idx)

    def _activate(self, logit, activation, layer_idx=-1):
        if activation == "sigmoid":
            return tf.nn.sigmoid(logit)
        elif activation == "softmax":
            return tf.nn.softmax(logit)
        elif activation == "relu":
            return tf.nn.relu(logit)
        elif activation == "tanh":
            return tf.nn.tanh(logit)
        elif activation == "elu":
            return tf.nn.elu(logit)
        elif activation == "identity":
            return tf.identity(logit)
        elif activation == 'dice':
            return dice(logit, name='dice_{}'.format(layer_idx))
        else:
            raise ValueError("this activations not defined {0}".format(activation))

    def _attention_fcn(self, query, user_embedding, max_iter):
        with tf.variable_scope("attention_fcn", initializer=tf.truncated_normal_initializer(stddev=0.01, seed=None)):

            for i in range(max_iter):
                n_item = self.news_attention(query, user_embedding, 40) # (None,40)
                query += n_item
                query = tf.nn.l2_normalize(query, axis=-1)

            attention_mat = tf.get_variable(
                name="attention_mat",
                shape=[user_embedding.shape.as_list()[-1], query.shape[1]],
                dtype=tf.float32,
            )
            sum_mat = tf.get_variable(
                name="sum_mat",
                shape=[40, 40],
                dtype=tf.float32,
            )

            att_inputs = tf.tensordot(user_embedding, attention_mat, [[2], [0]]) #(?, 50, 40)
            boolean_mask = tf.equal(self.mask, tf.ones_like(self.mask))

            query_size = query.shape[1]
            queries = tf.reshape(
                tf.tile(query, [1, att_inputs.shape[1]]), tf.shape(att_inputs)
            ) #(?, 50, 40)
            sum = tf.tensordot(tf.nn.tanh(queries + att_inputs), sum_mat, [[-1], [0]]) #(?, 50, 40)

            last_hidden_nn_layer = tf.concat(
                [queries*att_inputs, sum], -1
            ) #(?, 50, 80)

            att_fnc_output = self.new_fcn_net(
                last_hidden_nn_layer, att_fcn_layer_sizes, scope="att_fcn"
            ) #(?, 50, 1)
            att_fnc_output = tf.squeeze(att_fnc_output, -1) #(?, 50)
            mask_paddings = tf.ones_like(att_fnc_output) * (-(2 ** 32) + 1)
            att_weights = tf.nn.softmax(
                tf.where(boolean_mask, att_fnc_output, mask_paddings),
                name="att_weights",
            )
            output = user_embedding * tf.expand_dims(att_weights, -1) #(?, 50, 40)
            return output

    def short_attention_fcn(self, user_long, query, hist_input, rnn_outputs, att_size, max_iter):
        with tf.variable_scope("short_attention_fcn",
                               initializer=tf.truncated_normal_initializer(stddev=0.01, seed=None)):

            attention_mat = tf.get_variable(
                name="attention_mat",
                shape=[rnn_outputs.shape.as_list()[-1], 40],
                dtype=tf.float32,
            )
            att_inputs = tf.tensordot(rnn_outputs, attention_mat, [[2], [0]])  # att_inputs = [batch, hist_len, dim]

            short_term = user_cjpa(hist_input, user_long, att_size) #(?, 40)
            short_term = tf.reshape(
                tf.tile(short_term, [1, att_inputs.shape[1].value]), tf.shape(att_inputs)
            )  # (?, 50, 40)

            for i in range(max_iter):
                n_item = self.news_attention(query, short_term, 40) # (None,40)
                query += n_item
                query = tf.nn.l2_normalize(query, axis=-1)

            
            sum_mat = tf.get_variable(
                name="sum_mat",
                shape=[att_inputs.shape.as_list()[-1], att_inputs.shape.as_list()[-1]],
                dtype=tf.float32,
            )
            boolean_mask = tf.equal(self.mask, tf.ones_like(self.mask))

            query_size = query.shape[1]
            queries = tf.reshape(
                tf.tile(query, [1, att_inputs.shape[1].value]), tf.shape(att_inputs))  # (?, 50, 40)
            sum = tf.tensordot(tf.nn.tanh(queries + att_inputs), sum_mat, [[-1], [0]]) # (?, 50, 40)

            last_hidden_nn_layer = tf.concat([queries * att_inputs, sum], -1)

            att_fnc_output = self.new_fcn_net(
                last_hidden_nn_layer, att_fcn_layer_sizes, scope="att_fcn"
            )
            att_fnc_output = tf.squeeze(att_fnc_output, -1)
            mask_paddings = tf.ones_like(att_fnc_output) * (-(2 ** 32) + 1)
            att_weights = tf.nn.softmax(
                tf.where(boolean_mask, att_fnc_output, mask_paddings),
                name="att_weights",
            )
            output = att_inputs * tf.expand_dims(att_weights, -1)
            return output

    def _fcn_net(self, model_output, layer_sizes, i, scope):
        with tf.variable_scope(scope + str(i)):
            last_layer_size = model_output.shape[-1]
            layer_idx = 0
            hidden_nn_layers = []
            hidden_nn_layers.append(model_output)
            with tf.variable_scope("nn_part" + str(i),
                                   initializer=tf.truncated_normal_initializer(stddev=0.01, seed=None)) as scope:
                for idx, layer_size in enumerate(layer_sizes):
                    curr_w_nn_layer = tf.get_variable(
                        name="w_nn_layer" + str(i) + str(layer_idx),
                        shape=[last_layer_size, layer_size],
                        dtype=tf.float32,
                    )
                    curr_b_nn_layer = tf.get_variable(
                        name="b_nn_layer" + str(i) + str(layer_idx),
                        shape=[layer_size],
                        dtype=tf.float32,
                        initializer=tf.zeros_initializer(),
                    )

                    curr_hidden_nn_layer = (
                            tf.tensordot(
                                hidden_nn_layers[layer_idx], curr_w_nn_layer, axes=1
                            )
                            + curr_b_nn_layer
                    )

                    scope = "nn_part" + str(i) + str(idx)

                    curr_hidden_nn_layer = tf.layers.batch_normalization(
                        curr_hidden_nn_layer,
                        momentum=0.95,
                        epsilon=0.0001,
                        training=self.is_train_stage,
                    )

                    curr_hidden_nn_layer = self._active_layer(
                        logit=curr_hidden_nn_layer, activation='relu', layer_idx=idx
                    )
                    hidden_nn_layers.append(curr_hidden_nn_layer)
                    layer_idx += 1
                    last_layer_size = layer_size

                w_nn_output = tf.get_variable(
                    name="w_nn_output" + str(i),
                    shape=[last_layer_size, 1],
                    dtype=tf.float32,
                )
                b_nn_output = tf.get_variable(
                    name="b_nn_output" + str(i),
                    shape=[1],
                    dtype=tf.float32,
                    initializer=tf.zeros_initializer(),
                )

                nn_output = (
                        tf.tensordot(hidden_nn_layers[-1], w_nn_output, axes=1) + b_nn_output
                )
                self.logit = nn_output
                return nn_output

    def new_fcn_net(self, model_output, layer_sizes, scope):
        with tf.variable_scope(scope):
            last_layer_size = model_output.shape[-1]
            layer_idx = 0
            hidden_nn_layers = []
            hidden_nn_layers.append(model_output)
            with tf.variable_scope("nn_part",
                                   initializer=tf.truncated_normal_initializer(stddev=0.01,
                                                                               seed=None)) as scope:
                for idx, layer_size in enumerate(layer_sizes):
                    curr_w_nn_layer = tf.get_variable(
                        name="w_nn_layer" + str(layer_idx),
                        shape=[last_layer_size, layer_size],
                        dtype=tf.float32,
                    )
                    curr_b_nn_layer = tf.get_variable(
                        name="b_nn_layer" + str(layer_idx),
                        shape=[layer_size],
                        dtype=tf.float32,
                        initializer=tf.zeros_initializer(),
                    )

                    curr_hidden_nn_layer = (
                            tf.tensordot(
                                hidden_nn_layers[layer_idx], curr_w_nn_layer, axes=1
                            )
                            + curr_b_nn_layer
                    )

                    scope = "nn_part" + str(idx)

                    curr_hidden_nn_layer = tf.layers.batch_normalization(
                        curr_hidden_nn_layer,
                        momentum=0.95,
                        epsilon=0.0001,
                        training=self.is_train_stage,
                    )

                    curr_hidden_nn_layer = self._active_layer(
                        logit=curr_hidden_nn_layer, activation='relu', layer_idx=idx
                    )
                    hidden_nn_layers.append(curr_hidden_nn_layer)
                    layer_idx += 1
                    last_layer_size = layer_size

                w_nn_output = tf.get_variable(
                    name="w_nn_output",
                    shape=[last_layer_size, 1],
                    dtype=tf.float32,
                )
                b_nn_output = tf.get_variable(
                    name="b_nn_output",
                    shape=[1],
                    dtype=tf.float32,
                    initializer=tf.zeros_initializer(),
                )

                nn_output = (
                        tf.tensordot(hidden_nn_layers[-1], w_nn_output, axes=1) + b_nn_output
                )
                self.logit = nn_output
                return nn_output

    def load_model(self, model_path):
        if model_path is not None:
            act_path = model_path
        try:
            self.saver.restore(self.sess, act_path)
        except:
            raise IOError("Failed to find any matching files for {0}".format(act_path))


class BaseIterator(object):
    @abc.abstractmethod
    def parser_one_line(self, line):
        pass

    @abc.abstractmethod
    def load_data_from_file(self, infile):
        pass

    @abc.abstractmethod
    def _convert_data(self, labels, features):
        pass

    @abc.abstractmethod
    def gen_feed_dict(self, data_dict):
        pass


class SequentialIterator(BaseIterator):
    def __init__(self, graph):
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size
        self.iter_data = dict()

        self.graph = graph
        with self.graph.as_default():
            self.labels = tf.placeholder(tf.float32, [None, 1], name="label")
            self.users = tf.placeholder(tf.int32, [None], name="users")
            self.items = tf.placeholder(tf.int32, [None], name="items")
            self.cates = tf.placeholder(tf.int32, [None], name="cates")
            self.item_history = tf.placeholder(
                tf.int32, [None, self.max_seq_length], name="item_history"
            )
            self.item_cate_history = tf.placeholder(
                tf.int32, [None, self.max_seq_length], name="item_cate_history"
            )
            self.mask = tf.placeholder(
                tf.int32, [None, self.max_seq_length], name="mask"
            )
            self.time = tf.placeholder(tf.float32, [None], name="time")
            self.time_diff = tf.placeholder(
                tf.float32, [None, self.max_seq_length], name="time_diff"
            )
            self.time_from_first_action = tf.placeholder(
                tf.float32, [None, self.max_seq_length], name="time_from_first_action"
            )
            self.time_to_now = tf.placeholder(
                tf.float32, [None, self.max_seq_length], name="time_to_now"
            )
            self.attn_labels = tf.placeholder(tf.float32, [None, 1], name="attn_label")

    def parse_file(self, input_file):
        with open(input_file, "r") as f:
            lines = f.readlines()
        res = []
        for line in lines:
            if not line:
                continue
            res.append(self.parser_one_line(line))
        return res

    def get_item_history_sequence(self, item_history_words):
        item_history_sequence = []
        for item in item_history_words:
            item_history_sequence.append(
                load_dict(item_vocab)[item] if item in load_dict(item_vocab) else 0
            )
        return item_history_sequence

    def get_cate_history_sequence(self, cate_history_words):
        cate_history_sequence = []
        for cate in cate_history_words:
            cate_history_sequence.append(load_dict(cate_vocab)[cate] if cate in load_dict(cate_vocab) else 0)
        return cate_history_sequence

    def get_time_history_sequence(self, time_history_words):
        time_history_sequence = [float(i) for i in time_history_words]
        return time_history_sequence

    def parser_one_line(self, line):
        words = line.strip().split('\t')
        label = int(words[0])
        user_id = load_dict(user_vocab)[words[1]] if words[1] in load_dict(user_vocab) else 0
        item_id = load_dict(item_vocab)[words[2]] if words[2] in load_dict(item_vocab) else 0
        item_cate = load_dict(cate_vocab)[words[3]] if words[3] in load_dict(cate_vocab) else 0
        current_time = float(words[4])
        item_history_words = words[5].strip().split(",")
        cate_history_words = words[6].strip().split(",")
        item_history_sequence = self.get_item_history_sequence(item_history_words)
        cate_history_sequence = self.get_cate_history_sequence(cate_history_words)
        time_history_words = words[7].strip().split(",")
        time_history_sequence = self.get_time_history_sequence(time_history_words)
        time_range = 3600 * 24 / 1000

        time_diff = []
        for i in range(len(time_history_sequence) - 1):
            diff = (time_history_sequence[i + 1] - time_history_sequence[i]) / time_range
            diff = max(diff, 0.5)
            time_diff.append(diff)
        last_diff = (current_time - time_history_sequence[-1]) / time_range
        last_diff = max(last_diff, 0.5)
        time_diff.append(last_diff)
        time_diff = np.log(time_diff)

        first_time = time_history_sequence[0]
        time_from_first_action = [(t - first_time) / time_range for t in time_history_sequence[1:]]
        time_from_first_action = [max(t, 0.5) for t in time_from_first_action]
        last_diff = (current_time - first_time) / time_range
        last_diff = max(last_diff, 0.5)
        time_from_first_action.append(last_diff)
        time_from_first_action = np.log(time_from_first_action)

        time_to_now = [(current_time - t) / time_range for t in time_history_sequence]
        time_to_now = [max(t, 0.5) for t in time_to_now]
        time_to_now = np.log(time_to_now)

        return (
            label,
            user_id,
            item_id,
            item_cate,
            item_history_sequence,
            cate_history_sequence,
            current_time,
            time_diff,
            time_from_first_action,
            time_to_now,
        )

    def _convert_data(self,
                      label_list,
                      user_list,
                      item_list,
                      item_cate_list,
                      item_history_batch,
                      item_cate_history_batch,
                      time_list,
                      time_diff_list,
                      time_from_first_action_list,
                      time_to_now_list,
                      batch_num_ngs,
                      ):
        if batch_num_ngs:
            instance_cnt = len(label_list)
            if instance_cnt < 5:
                return

            label_list_all = []
            item_list_all = []
            item_cate_list_all = []
            attn_label_list_all = []

            user_list_all = np.asarray(
                [[user] * (batch_num_ngs + 1) for user in user_list], dtype=np.int32
            ).flatten()
            time_list_all = np.asarray(
                [[t] * (batch_num_ngs + 1) for t in time_list], dtype=np.float32
            ).flatten()

            history_lengths = [len(item_history_batch[i]) for i in range(instance_cnt)]
            max_seq_length_batch = self.max_seq_length
            item_history_batch_all = np.zeros(
                (instance_cnt * (batch_num_ngs + 1), max_seq_length_batch)
            ).astype("int32")
            item_cate_history_batch_all = np.zeros(
                (instance_cnt * (batch_num_ngs + 1), max_seq_length_batch)
            ).astype("int32")
            time_diff_batch = np.zeros(
                (instance_cnt * (batch_num_ngs + 1), max_seq_length_batch)
            ).astype("float32")
            time_from_first_action_batch = np.zeros(
                (instance_cnt * (batch_num_ngs + 1), max_seq_length_batch)
            ).astype("float32")
            time_to_now_batch = np.zeros(
                (instance_cnt * (batch_num_ngs + 1), max_seq_length_batch)
            ).astype("float32")
            mask = np.zeros(
                (instance_cnt * (1 + batch_num_ngs), max_seq_length_batch)
            ).astype("float32")

            for i in range(instance_cnt):
                this_length = min(history_lengths[i], max_seq_length_batch)
                for index in range(batch_num_ngs + 1):
                    item_history_batch_all[
                        i * (batch_num_ngs + 1) + index, :this_length
                    ] = np.asarray(item_history_batch[i][-this_length:], dtype=np.int32) #正负样本一样
                    item_cate_history_batch_all[
                        i * (batch_num_ngs + 1) + index, :this_length
                    ] = np.asarray(
                        item_cate_history_batch[i][-this_length:], dtype=np.int32
                    )
                    mask[i * (batch_num_ngs + 1) + index, :this_length] = 1.0
                    time_diff_batch[
                        i * (batch_num_ngs + 1) + index, :this_length
                    ] = np.asarray(time_diff_list[i][-this_length:], dtype=np.float32)
                    time_from_first_action_batch[
                        i * (batch_num_ngs + 1) + index, :this_length
                    ] = np.asarray(
                        time_from_first_action_list[i][-this_length:], dtype=np.float32
                    )
                    time_to_now_batch[
                        i * (batch_num_ngs + 1) + index, :this_length
                    ] = np.asarray(time_to_now_list[i][-this_length:], dtype=np.float32)

            for i in range(instance_cnt):
                this_length = min(history_lengths[i], max_seq_length_batch)
                item_cate_history = np.asarray(item_cate_history_batch[i][-this_length:], dtype=np.int32)
                positive_item = item_list[i]
                label_list_all.append(1)
                item_list_all.append(positive_item)
                item_cate_list_all.append(item_cate_list[i])
                attn_label = (item_cate_history == item_cate_list[i]).sum()/this_length
                attn_label_list_all.append(attn_label)
                count = 0
                while batch_num_ngs:
                    random_value = random.randint(0, instance_cnt - 1)
                    negative_item = item_list[random_value]
                    if negative_item == positive_item:
                        continue
                    label_list_all.append(0)
                    item_list_all.append(negative_item)
                    item_cate_list_all.append(item_cate_list[random_value])
                    attn_label = (item_cate_history == item_cate_list[random_value]).sum()/this_length
                    attn_label_list_all.append(attn_label)
                    count += 1
                    if count == batch_num_ngs:
                        break

            res = {}
            res["labels"] = np.asarray(label_list_all, dtype=np.float32).reshape(-1, 1)
            res["attn_labels"] = np.asarray(attn_label_list_all, dtype=np.float32).reshape(-1, 1)
            res["users"] = user_list_all
            res["items"] = np.asarray(item_list_all, dtype=np.int32)
            res["cates"] = np.asarray(item_cate_list_all, dtype=np.int32)
            res["item_history"] = item_history_batch_all
            res["item_cate_history"] = item_cate_history_batch_all
            res["mask"] = mask
            res["time"] = time_list_all
            res["time_diff"] = time_diff_batch
            res["time_from_first_action"] = time_from_first_action_batch
            res["time_to_now"] = time_to_now_batch
            return res

        else:
            instance_cnt = len(label_list)
            history_lengths = [len(item_history_batch[i]) for i in range(instance_cnt)]
            max_seq_length_batch = self.max_seq_length
            item_history_batch_all = np.zeros(
                (instance_cnt, max_seq_length_batch)
            ).astype("int32")
            item_cate_history_batch_all = np.zeros(
                (instance_cnt, max_seq_length_batch)
            ).astype("int32")
            time_diff_batch = np.zeros((instance_cnt, max_seq_length_batch)).astype(
                "float32"
            )
            time_from_first_action_batch = np.zeros(
                (instance_cnt, max_seq_length_batch)
            ).astype("float32")
            time_to_now_batch = np.zeros((instance_cnt, max_seq_length_batch)).astype(
                "float32"
            )
            mask = np.zeros((instance_cnt, max_seq_length_batch)).astype("float32")

            attn_label_list = []
            for i in range(instance_cnt):
                this_length = min(history_lengths[i], max_seq_length_batch)
                item_history_batch_all[i, :this_length] = item_history_batch[i][
                    -this_length:
                ]
                item_cate_history_batch_all[i, :this_length] = item_cate_history_batch[
                    i
                ][-this_length:]
                item_cate_history = np.asarray(item_cate_history_batch[i][-this_length:], dtype=np.int32)
                attn_label = (item_cate_history == item_cate_list[i]).sum()/this_length
                attn_label_list.append(attn_label)
                mask[i, :this_length] = 1.0
                time_diff_batch[i, :this_length] = time_diff_list[i][-this_length:]
                time_from_first_action_batch[
                    i, :this_length
                ] = time_from_first_action_list[i][-this_length:]
                time_to_now_batch[i, :this_length] = time_to_now_list[i][-this_length:]

            res = {}
            res["labels"] = np.asarray(label_list, dtype=np.float32).reshape(-1, 1)
            res["attn_labels"] = np.asarray(attn_label_list, dtype=np.float32).reshape(-1, 1)
            res["users"] = np.asarray(user_list, dtype=np.float32)
            res["items"] = np.asarray(item_list, dtype=np.int32)
            res["cates"] = np.asarray(item_cate_list, dtype=np.int32)
            res["item_history"] = item_history_batch_all
            res["item_cate_history"] = item_cate_history_batch_all
            res["mask"] = mask
            res["time"] = np.asarray(time_list, dtype=np.float32)
            res["time_diff"] = time_diff_batch
            res["time_from_first_action"] = time_from_first_action_batch
            res["time_to_now"] = time_to_now_batch
            return res

    def load_data_from_file(self, infile, batch_num_ngs, min_seq_length):
        label_list = []
        user_list = []
        item_list = []
        item_cate_list = []
        item_history_batch = []
        item_cate_history_batch = []
        time_list = []
        time_diff_list = []
        time_from_first_action_list = []
        time_to_now_list = []

        cnt = 0

        if infile not in self.iter_data:
            lines = self.parse_file(infile)
            self.iter_data[infile] = lines
        else:
            lines = self.iter_data[infile]

        if batch_num_ngs > 0:
            random.shuffle(lines)

        for line in lines:
            if not line:
                continue

            (
                label,
                user_id,
                item_id,
                item_cate,
                item_history_sequence,
                item_cate_history_sequence,
                current_time,
                time_diff,
                time_from_first_action,
                time_to_now,
            ) = line
            if len(item_history_sequence) < min_seq_length:
                continue

            label_list.append(label)
            user_list.append(user_id)
            item_list.append(item_id)
            item_cate_list.append(item_cate)
            item_history_batch.append(item_history_sequence)
            item_cate_history_batch.append(item_cate_history_sequence)
            time_list.append(current_time)
            time_diff_list.append(time_diff)
            time_from_first_action_list.append(time_from_first_action)
            time_to_now_list.append(time_to_now)

            cnt += 1
            if cnt == batch_size:
                res = self._convert_data(
                    label_list,
                    user_list,
                    item_list,
                    item_cate_list,
                    item_history_batch,
                    item_cate_history_batch,
                    time_list,
                    time_diff_list,
                    time_from_first_action_list,
                    time_to_now_list,
                    batch_num_ngs,
                )
                #batch_input = self.gen_feed_dict(res)
                yield res
                label_list = []
                user_list = []
                item_list = []
                item_cate_list = []
                item_history_batch = []
                item_cate_history_batch = []
                time_list = []
                time_diff_list = []
                time_from_first_action_list = []
                time_to_now_list = []
                cnt = 0
        if cnt > 0:
            res = self._convert_data(
                label_list,
                user_list,
                item_list,
                item_cate_list,
                item_history_batch,
                item_cate_history_batch,
                time_list,
                time_diff_list,
                time_from_first_action_list,
                time_to_now_list,
                batch_num_ngs,
            )
            #batch_input = self.gen_feed_dict(res)
            yield res


att_fcn_layer_sizes = [80, 40]
max_seq_length = 50
batch_size = 200
user_embedding_dim = 40
item_embedding_dim = 32
cate_embedding_dim = 8
max_iter = 12
att_size = 80
RANDOM_SEED = None
data_path = '/nfsshare/home/chenzixuan/'
save_path = os.path.join(data_path, 'our_new_13')
model_path = os.path.join(save_path, "model/")

train_file = os.path.join(data_path, r'sub_train_data')
valid_file = os.path.join(data_path, r'sub_valid_data')
test_file = os.path.join(data_path, r'sub_test_data')

user_vocab = os.path.join(data_path, r'user_vocab.pkl')
item_vocab = os.path.join(data_path, r'item_vocab.pkl')
cate_vocab = os.path.join(data_path, r'category_vocab.pkl')

eval_metric = 'wauc'

input_creator = SequentialIterator
model = BaseModel(input_creator, RANDOM_SEED)
print('11111')
#model = model.fit(train_file, valid_file, train_num_ngs=4, valid_num_ngs=0, eval_metric=eval_metric)

ckpt_path = tf.train.latest_checkpoint(model_path)
model.load_model(ckpt_path)
res = model.run_weighted_eval(test_file, num_ngs=0, fla=False)
print(res)
