from tensorflow.keras.callbacks import EarlyStopping,ReduceLROnPlateau

pocket = EarlyStopping(monitor='val_exact_matched_accuracy', min_delta=0.001,
                       patience=10, verbose=1, mode='max', 
                       restore_best_weights = True)
reduce_lr = ReduceLROnPlateau(monitor = 'val_accuracy', factor = 0.3, patience = 2, min_delta = 0.001,
                              mode='auto',verbose=1)
history=model.fit(xtrain,ytrain,epochs=20,batch_size=256,validation_data=(xtest,ytest),callbacks=[pocket,reduce_lr])
# history = pd.DataFrame(history.history)

import matplotlib.pyplot as plt
fig, ax = plt.subplots(2,1)
ax[0].plot(history.history['loss'], color='b',linestyle = 'dashdot', label="Training loss")
ax[0].plot(history.history['val_loss'], color='r', linestyle = 'dashdot',label="validation loss",axes =ax[0])
legend = ax[0].legend(loc='best', shadow=True)

ax[1].plot(history.history['accuracy'], color='b',linestyle = 'dashdot' ,label="Training accuracy")
ax[1].plot(history.history['val_accuracy'], color='r',linestyle = 'dashdot',label="Validation accuracy")
legend = ax[1].legend(loc='best', shadow=True)

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import numpy as np

# Assuming 'model' is your Keras functional model
x_train_each_class = ['weakup','oneset']

# Convert multiclass labels to one-hot encoding
y_test_bin = label_binarize(ytest, classes=np.arange(2))

# Get predicted probabilities for each class
y_pred_proba = model.predict(xtest)

# Calculate ROC curve and AUC for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(2):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot ROC curves for each class
plt.figure(figsize=(8, 6))
for i in range(2):
    plt.plot(fpr[i], tpr[i], label=f'Class {x_train_each_class[i]} (AUC= {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curves')
plt.legend(loc="lower right")
plt.show()

y_pred = model.predict(xtest)
y_pred_test = [np.argmax(i) for i in y_pred]
print("Confusion Matrix:\n",confusion_matrix(ytest,y_pred_test))
print()


print("Classification Report:\n",classification_report(ytest,y_pred_test))

print("Confusion Matrix:\n",confusion_matrix(ytest,y_pred_test))
print()


print("Classification Report:\n",classification_report(ytest,y_pred_test))
import seaborn as sns
labels=['Oneset','Weakup']
cm_data=confusion_matrix(ytest,y_pred_test)
cm = pd.DataFrame(cm_data, columns=labels, index =labels)
cm.index.name = 'Actual'
cm.columns.name = 'Predicted'
plt.figure(figsize = (5,5))
plt.title('Confusion Matrix', fontsize = 20)
sns.set(font_scale=1.2)
ax = sns.heatmap(cm, cbar=False, cmap="crest", annot=True, annot_kws={"size": 16}, fmt='g')