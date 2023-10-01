Abschlussaufgabe Python Kurs SS2023

Entwickeln Sie Neuronale Netze mit Keras zur Klassifizierung von Bildern aus dem CIFAR-10-Datensatz und dem CIFAR-100-Datensatz. 
Ihre Modelle sollen in der Lage sein, zwischen verschiedenen Klassen des Datensätze zu unterscheiden. Schreiben Sie ALLES in das Notebook! Auch wenn Sie Informationen aus dem Internet verwenden

Anforderungen:

    Daten laden und vorbereiten:
        Laden Sie den CIFAR-10-Datensatz bzw. den CIFAR-100-Datensatz und analysieren Sie den Datensatz bzgl Datenmenge, -format, Labeln, ... Behandeln Sie den Datensatz wie eine Blackbox und versuchen alles ohne fremde Hilfe herauszufinden
        Führen Sie eine geeignete Vorverarbeitung durch.
        Visualisieren Sie zudem das erste Bild jeder Klasse des CIFAR-10-Datensatzes mit dem passenden Label

    Modellarchitektur: Entwerfen Sie mehrere Modellarchitekturen. Sie können verschiedene Arten von Schichten verwenden, wie Conv2D, MaxPooling2D, Dropout, Flatten und Dense.

    Modell kompilieren: Wählen Sie einen geeigneten Optimierer und eine Verlustfunktion für die Klassifizierungsaufgabe.

    Modell trainieren: Trainieren Sie Ihr Modell auf den Trainingsdaten und validieren Sie es auf den Validierungsdaten.

    Modell evaluieren: Evaluieren Sie Ihr Modell auf den Testdaten und berichten Sie über die Genauigkeit.

    Visualisierung: Visualisieren Sie den Trainings- und Validierungsverlust sowie die Genauigkeit im Laufe der Epochen.

    Können Sie beim CIFAR-100-Datensatz genauso wie beim CIFAR-10-Datensatz vorgehen? Schreiben Sie ALLE Vermutungen und Informationen auf!

CIFAR-10________________________________________________________________________________________________________________________________
```python
# Laden der benötigten Libraries
import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

# Laden & Untersuchen der Datensätze
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

    # Samples zählen
print(len(train_labels), len(test_labels))

    # Bildformat ausgeben
image_size = np.asarray([train_images.shape[1], train_images.shape[2], train_images.shape[3]])
print(image_size)

    # Labels ausgeben (Ergebnis: 10 Klassen als Integers, scheinbar müssen die Bedeutungen manuell codiert werden)
unique_labels = np.unique(train_labels)
print(unique_labels)

    # Visualisierung des ersten Bilds jeder Klasse (→ Ansatz: Label-Array nach Klassennamen durchsuchen, zugehörigen Index rückgeben & diesen im Bild-Array adressieren)
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'] # Diese Liste wurde im Nachhinein angepasst, so dass der Plot stimmt
    
plt.figure(figsize=(10,5))
for j in range(len(unique_labels)): # Iterieren jedes Elements aus der Liste der einzigartigen Integers
    plt.subplot(2,5,j+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    for i in range(len(train_labels)): # Iterieren der Label-Liste bis Listenwert dem aktuellen Element der einzigartigen Labels entspricht
        if train_labels[i][0] == j: # Zweiter Index ist wichtig, da nur damit der erste/einzige Wert des nested arrays adressiert wird
            plt.imshow(train_images[i])
            plt.xlabel(class_names[train_labels[i][0]])
            break
plt.show()

# Vorverarbeitung bzw. Skalieren der Bilder für das neuronale Netzwerk
train_images = train_images/255
test_images = test_images/255

# Aufbau der Modellarchitekturen (Vorteil von Conv2D-Nets ggü Dense-Layer-Nets: Translationsinvarianz & räumlich-hierarchische Einbettung) (MoxPooling2D: Downsampling der feature-maps bewirkt Generalisierung)
model1 = models.Sequential()
model1.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model1.add(layers.MaxPooling2D((2, 2)))
model1.add(layers.Conv2D(64, (3, 3), activation='relu'))
model1.add(layers.MaxPooling2D((2, 2)))
model1.add(layers.Conv2D(64, (3, 3), activation='relu'))
model1.add(layers.Flatten())
model1.add(layers.Dense(64, activation='relu'))
model1.add(layers.Dense(10))

model2 = models.Sequential()
model2.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model2.add(layers.MaxPooling2D((2, 2)))
model2.add(layers.Conv2D(64, (3, 3), activation='relu'))
model2.add(layers.MaxPooling2D((2, 2)))
model2.add(layers.Conv2D(64, (3, 3), activation='relu'))
model2.add(layers.MaxPooling2D((2, 2))) # eigentlich will man ein noch größeres Modell; leider kann man nicht noch mehr Conv2D nutzen, da feature-maps irgendwann nicht mehr halbierbar
model2.add(layers.Flatten())
model2.add(layers.Dense(64, activation='relu'))
model2.add(layers.Dense(10))

    # Übersicht der Modelle plotten
model1.summary()
model2.summary()

# Kompilieren & Trainieren der Modelle nach Chollet Kap-5.1 (SparseCategoricalCrossentropy statt CategoricalCrossentropy, da kein One-Hot-Encoding, sondern Codierung der Labels als Integer-Tensor)
model1.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
              metrics=['accuracy'])

history1 = model1.fit(train_images, train_labels, epochs=10, 
                    validation_data=(test_images, test_labels))

model2.compile(optimizer='rmsprop',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
              metrics=['accuracy'])

history2 = model2.fit(train_images, train_labels, epochs=15, 
                    validation_data=(test_images, test_labels))

# Evaluation & Visualisierung der Modelle anhand der Testdaten
# Modell1
plt.figure(figsize=(20,5))
    # Verlust
plt.subplot(1,2,1)
plt.plot(history1.history['loss'], label='loss')
plt.plot(history1.history['val_loss'], label = 'val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
#plt.ylim([0.5, 1])
plt.legend(loc='lower right')
test_loss1, test_acc1 = model1.evaluate(test_images,  test_labels, verbose=2) # verbose {0,1,2} bestimmt nur die Darstellung des Epochen-Fortschritts
    # Genauigkeit
plt.subplot(1,2,2)
plt.plot(history1.history['accuracy'], label='accuracy')
plt.plot(history1.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
#plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.show()

print(test_acc1)

# Modell2
plt.figure(figsize=(20,5))
    # Verlust
plt.subplot(1,2,1)
plt.plot(history2.history['loss'], label='loss')
plt.plot(history2.history['val_loss'], label = 'val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
#plt.ylim([0.5, 1])
plt.legend(loc='lower right')
test_loss2, test_acc2 = model2.evaluate(test_images,  test_labels, verbose=2) # verbose {0,1,2} bestimmt nur die Darstellung des Epochen-Fortschritts
    # Genauigkeit
plt.subplot(1,2,2)
plt.plot(history2.history['accuracy'], label='accuracy')
plt.plot(history2.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
#plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.show()

print(test_acc2)
```

CIFAR-100_____________________________________________________________________________________________________________________
```python
# Was ist anders als bei CIFAR-10: Es ist ein Multilabel-Datensatz, womit eine andere Aktivierungs- (sigmoid) & Loss-Funktion (binary_crossentropy) benutzt werden kann (siehe Chollet S.343)
# Dazu benötigt man jedoch eine k-hot-Codierung
# Ferner muss der Output der letzten Schicht von 10 auf 100 erhöht werden

# Laden der benötigten Libraries
import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models, losses, optimizers
import matplotlib.pyplot as plt

#batch_size = 50
#img_width, img_height, img_num_channels = 32, 32, 3
#loss_function = sparse_categorical_crossentropy
#no_classes = 100
#no_epochs = 30
#optimizer = Adam()
#validation_split = 0.2
#verbosity = 1

# Laden & Untersuchen der Datensätze (Hier wählt man Class oder Superclass, merkwürdig nur, dass das scheinbar ohne Auswirkungen auf die letzte Dense-Layer bleibt)
(train_images, train_labels), (test_images, test_labels) = datasets.cifar100.load_data(label_mode='fine') #(label_mode='coarse')

    # Samples zählen
print(len(train_labels), len(test_labels))

    # Bildformat ausgeben
image_size = np.asarray([train_images.shape[1], train_images.shape[2], train_images.shape[3]])
print(image_size)

    # Labels ausgeben (Ergebnis: 100 Klassen als Integers bei label_mode='fine', aber 20 bei label_mode='coarse')
unique_labels = np.unique(train_labels)
print(unique_labels)

# Determine shape of the data
#input_shape = (img_width, img_height, img_num_channels)

# Parse numbers as floats
train_images = train_images.astype('float32')
test_images = test_images.astype('float32')

# Vorverarbeitung bzw. Skalieren der Bilder für das neuronale Netzwerk
train_images = train_images/255
test_images = test_images/255

# Aufbau der Modellarchitektur
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(100, activation='softmax'))

    # Übersicht der Modelle plotten
model.summary()

# Kompilieren des Modells
model.compile(loss=losses.SparseCategoricalCrossentropy(), optimizer='adam', metrics=['accuracy'])

# Trainieren des Modells
history = model.fit(train_images, train_labels, batch_size=50, epochs=30, verbose=1, validation_split=0.2)

# Evaluation & Visualisierung der Modelle anhand der Testdaten
    # Kennzahlen ausgeben
score = model.evaluate(test_images, test_labels, verbose=0) # verbose {0,1,2} bestimmt nur die Darstellung des Epochen-Fortschritts
print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')

plt.figure(figsize=(20,5))
    # Verlust
plt.subplot(1,2,1)
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label = 'val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='lower right')
    # Genauigkeit
plt.subplot(1,2,2)
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()
```
