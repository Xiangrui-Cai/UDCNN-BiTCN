# %%
import os
import time
from sklearn.utils import shuffle
import pandas as pd
import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay
from sklearn.metrics import cohen_kappa_score
import tensorflow as tf
import UDCNN_BITCN_tensorflow
from preprocess import get_data
import tensorflow_probability as tfp

def mixup_batch(x, y, alpha=1.0):
    lam = tfp.distributions.Beta(alpha, alpha).sample(1)[0]
    # 生成随机排列的索引
    batch_size = tf.shape(x)[0]
    # tf.print("Batch size:", batch_size)  # 在 TensorFlow 图执行期间打印
    indices = tf.random.shuffle(tf.range(batch_size))
    # 混合数据
    x_mixed = lam * x + (1 - lam) * tf.gather(x, indices)
    y_mixed = lam * y + (1 - lam) * tf.gather(y, indices)

    return x_mixed, y_mixed

class CustomCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % 30 == 0:
            print(f"Epoch {epoch+1}, Validation Loss: {logs['val_loss']:.4f}")
            # log_write.write(f"Epoch {epoch+1}, Validation Loss: {logs['val_loss']:.4f}\n")

def train(dataset_conf, train_conf, results_path):
    in_exp = time.time()
    best_models = open(os.path.join(results_path, "best models.txt"), "w")
    log_write = open(os.path.join(results_path, "log.txt"), "w")

    # 定义DataFrame列：添加 'Early_Stop_Epoch'
    columns = ['Subject', 'Best_Run', 'Accuracy', 'Avg_Accuracy', 'Std_Accuracy',
               'TP', 'FP', 'TN', 'FN', 'Early_Stop_Epoch']
    results_df = pd.DataFrame(columns=columns)

    n_sub = dataset_conf.get('n_sub')
    n_train = train_conf.get('n_train')

    # 初始化acc数组（用于统计）
    acc = np.zeros((n_sub, n_train))
    all_metrics = {}  # 存储每个subject的最佳TP/FP/TN/FN比率

    for sub in range(n_sub):

        in_sub = time.time()
        print('\nTraining on subject ', sub + 1)
        log_write.write('\nTraining on subject  ' + str(sub + 1) + '\n')

        X_train, _, y_train_onehot, X_test, _, y_test_onehot = get_data(dataset_conf.get('data_path'), sub,
                                                                        dataset_conf.get('isStandard'))
        X_train, y_train_onehot = shuffle(X_train, y_train_onehot)

        X_train = X_train.astype(np.float32)
        y_train_onehot = y_train_onehot.astype(np.float32)

        # 用于存储每次运行的acc和metrics
        acc_list = []
        es_list = []  # 用于记录early stopping的epoch
        metrics_list = []

        for train_idx in range(n_train):
            in_run = time.time()
            filepath = os.path.join(results_path, f'saved models/run-{train_idx + 1}',
                                    f'subject-{sub + 1}.h5')
            os.makedirs(os.path.dirname(filepath), exist_ok=True)

            model = getModel(train_conf.get('model'))
            model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=train_conf.get('lr')),
                          metrics=['accuracy'])

            callbacks = [
                tf.keras.callbacks.ModelCheckpoint(filepath, monitor='val_accuracy', verbose=0, save_best_only=True,
                                                save_weights_only=True, mode='max'),
                tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=train_conf.get('patience'), mode='max', verbose=1),
                CustomCallback()  # 添加自定义回调
            ]

            # 创建动态mixup数据管道
            train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train_onehot))
            train_dataset = train_dataset.shuffle(buffer_size=1024, reshuffle_each_iteration=True)
            train_dataset = train_dataset.batch(train_conf.get('batch_size'))
            train_dataset = train_dataset.map(
                lambda x, y: mixup_batch(x, y, alpha=0.2),
                num_parallel_calls=tf.data.AUTOTUNE
            ).prefetch(tf.data.AUTOTUNE)

            # 训练模型
            history = model.fit(
                train_dataset,
                validation_data=(X_test, y_test_onehot),
                epochs=train_conf.get('epochs'),
                callbacks=callbacks,
                verbose=0
            )

            model.load_weights(filepath)
            y_pred = model.predict(X_test).argmax(axis=-1)
            labels = y_test_onehot.argmax(axis=-1)

            current_acc = accuracy_score(labels, y_pred)
            acc[sub, train_idx] = current_acc
            acc_list.append(current_acc)

            # 获取 early stopping 的 epoch
            best_val_epoch = np.argmax(history.history['val_accuracy']) + 1
            early_stop_epoch = best_val_epoch + train_conf.get('patience')
            es_list.append(early_stop_epoch)

            # 计算混淆矩阵并转换为百分比
            tn, fp, fn, tp = confusion_matrix(labels, y_pred).ravel()
            metrics_list.append((tn, fp, fn, tp))

            out_run = time.time()
            info = f'Subject: {sub + 1}   Train no. {train_idx + 1}   Time: {(out_run - in_run) / 60:.1f} m   '
            info += f'Test_acc: {current_acc:.4f}   Early Stop Epoch: {early_stop_epoch}'
            print(info)
            log_write.write(info + '\n')

        # 找出最佳准确率的那次训练
        best_run_idx = np.argmax(acc_list)
        best_acc = acc_list[best_run_idx]
        best_es_epoch = es_list[best_run_idx]
        tn, fp, fn, tp = metrics_list[best_run_idx]

        avg_acc = np.mean(acc_list)
        std_acc = np.std(acc_list)

        # 写入 best models.txt
        best_filepath = f'/saved models/run-{best_run_idx + 1}/subject-{sub + 1}.h5\n'
        best_models.write(best_filepath)

        # 输出当前subject的结果
        out_sub = time.time()
        info = f'----------\nSubject: {sub + 1}   best_run: {best_run_idx + 1}   Time: {(out_sub - in_sub) / 60:.1f} m\n'
        info += f'acc: {best_acc:.4f}   avg_acc: {avg_acc:.4f} ± {std_acc:.4f}\n'
        info += f'TN: {tn:.4f}, FP: {fp:.4f}, FN: {fn:.4f}, TP: {tp:.4f}\n'
        info += f'Early Stop Epoch: {best_es_epoch}\n'
        info += '----------'
        print(info)
        log_write.write(info + '\n')

        # 添加到DataFrame
        new_row = {
            'Subject': sub + 1,
            'Best_Run': best_run_idx + 1,
            'Accuracy': best_acc,
            'Avg_Accuracy': avg_acc,
            'Std_Accuracy': std_acc,
            'TP': tp,
            'FP': fp,
            'TN': tn,
            'FN': fn,
            'Early_Stop_Epoch': best_es_epoch
        }
        results_df = results_df._append(new_row, ignore_index=True)

    # 保存所有结果到Excel
    excel_path = os.path.join(results_path, "results.xlsx")
    results_df.to_excel(excel_path, index=False)
    print(f"\nResults saved to: {excel_path}")

    # 保存原始acc数组
    np.savez(os.path.join(results_path, "perf_allRuns.npz"), acc=acc)

    out_exp = time.time()
    total_time = (out_exp - in_exp) / 3600
    print(f'\nTotal training time: {total_time:.2f} hours')
    log_write.write(f'\nTotal training time: {total_time:.2f} hours\n')

    # 关闭文件
    best_models.close()
    log_write.close()

# %%
def getModel(model_name):
    # Select the model
    if (model_name == 'CNNNet'):
        model = UDCNN_BITCN_tensorflow.UDCNN_BiTCN(
            # Dataset parameters
            n_classes=2,
            in_chans=32,
            in_samples=1500,
            eegn_F1=64,
            eegn_D=2,
            eegn_kernelSize=64,
            eegn_dropout=0.3,
        )
    else:
        raise Exception("'{}' model is not supported yet!".format(model_name))

    return model

def run():
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    data_path = r'raw data//'
    results_path = os.getcwd() + r"/results"
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    # Set dataset paramters
    dataset_conf = {'n_classes': 2, 'n_sub': 97, 'n_channels': 32, 'data_path': data_path,
                    'isStandard': True, 'LOSO': False}

    # Set training hyperparamters
    train_conf = {'batch_size': 10, 'epochs': 1000, 'patience': 300, 'lr': 0.0009,
                  'LearnCurves': True, 'n_train': 10, 'model': 'CNNNet'}
    
    train(dataset_conf, train_conf, results_path)

# %%
if __name__ == "__main__":
    run()



