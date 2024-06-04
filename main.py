import os
import cv2
import pickle
import numpy as np
from PIL import Image
import tensorflow as tf
import streamlit as st

from sklearn.svm import SVC
from keras_facenet import FaceNet
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


def extractFaces(file_path, output_image_size=(160, 160)):
    # Load the HaarCascade model XLM file
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Load the image
    img = cv2.imread(file_path)
    # Convert the image into a numpy array
    img = np.asarray(img)

    # Detect the faces using the face cascade classifier
    faces = face_cascade.detectMultiScale(img, scaleFactor=float(1.2), minNeighbors=10)

    # Loop through all the faces detected by the HaarCascade Model
    for (x, y, w, h) in faces:
        # Extract the region of the image corresponding to the detected face
        detected_face = img[y:y + h, x:x + w]

        try:
            # Resize the image according to the requirements
            detected_face = cv2.resize(detected_face, dsize=output_image_size)

        except cv2.error as e:
            raise ValueError(f"Error resizing image: {e}")

        return detected_face

    return None


def load_faces(folder_path):
    # Initialize an empty list to store face images
    faces = {}
    counter = 0
    # Iterate over all files in the given folder
    for file in os.listdir(folder_path):

        # Construct the full path to the file
        path = folder_path + file

        # Check if the current item is a file
        if os.path.isfile(path):
            # Extract faces from the image at the given path
            face_img = extractFaces(path)

            # If face extraction was successful (i.e., a face was found), add it to the faces dictionary
            if face_img is not None:
                # file corresponds the name of the face owner
                faces[file] = face_img
                counter += 1

        if counter >= 40:
            # We only need 40 images per person
            break

    # Return the dictionary celebrities and their respective faces
    return faces


def load_dataset():
    # Parent directory path
    parent_directory = 'Faces-Dataset/'

    # Initialize empty lists to store face images (X) and corresponding labels (Y)
    # X and Y store objects of type numpy.array
    X = []
    Y = []

    # Iterate over each sub-directory in the given folder
    for sub_dir in os.listdir(parent_directory):
        # Construct the full path to the sub-directory
        path = parent_directory + sub_dir + '/'

        # Load faces from the sub-directory
        face_dict = load_faces(path)

        # Load the face images
        detected_faces = face_dict.values()

        # Load the labels for the detected faces
        labels = [sub_dir for _ in range(len(face_dict.keys()))]

        # Extend the face images list (X) with the detected faces
        X.extend(detected_faces)

        # Extend the labels list (Y) with the created labels
        Y.extend(labels)

    # Convert the lists to numpy arrays for efficient numerical operations
    X_final = np.asarray(X)
    Y_final = np.asarray(Y)

    # Split the dataset into Train & Test sets
    X_train, X_test, Y_train, Y_test = train_test_split(X_final, Y_final, test_size=0.20, random_state=17)

    # Save the arrays to an NPZ file
    np.savez('generated_images.npz', array1=X_train, array2=X_test, array3=Y_train, array4=Y_test)


def get_embedding(embedder,img):

    # Converting the image to the LUV channel, providing more distinction between the different features of the face
    face_img = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)

    # Expanding the dimensions of the face_img array by adding an extra dimension at the beginning.
    # This is done to match the input shape expected by neural networks, where the first dimension represents the batch size.
    face_img = np.expand_dims(face_img, axis=0)

    # Converting the data type of the face_img array to float32.
    # This is a common practice in machine learning to ensure numerical precision and compatibility with certain neural network libraries.
    face_img = face_img.astype('float32')

    # Generate embeddings of the image
    yhat = embedder.embeddings(face_img)

    return yhat[0]


def get_all_embeddings():
    # Load the embedder
    embedder = FaceNet()

    # Load the .npz file
    data = np.load('generated_images.npz')

    X_train = data['array1']
    X_test = data['array2']
    Y_train = data['array3']
    Y_test = data['array4']
    
    EMBEDDED_TRAIN_SET = []
    EMBEDDED_TEST_SET = []

    for img in X_train:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        EMBEDDED_TRAIN_SET.append(get_embedding(embedder, img))

    for img in X_test:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        EMBEDDED_TEST_SET.append(get_embedding(embedder, img))

    X_train_embeddings = np.asarray(EMBEDDED_TRAIN_SET)
    X_test_embeddings = np.asarray(EMBEDDED_TEST_SET)

    # Save the arrays to npz file
    np.savez('embeddings.npz', array1=X_train_embeddings, array2=X_test_embeddings, array3=Y_train, array4=Y_test)

def train_svm():
    # Load the Train & Test sets
    data = np.load('embeddings.npz')

    X_train = data['array1']
    X_test = data['array2']
    Y_train = data['array3']
    Y_test = data['array4']

    # Encode data
    encoder = LabelEncoder()
    encoder.fit(Y_train)
    Y_train = encoder.transform(Y_train)
    Y_test = encoder.transform(Y_test)
    
    # Initialize SVM model
    model = SVC(probability=True)

    # Set grid search
    # Create a parameter grid: map the parameter names to the values that should be searched
    param_grid = {
        'C': [0.1, 1, 10, 100, 1000],
        'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
        'kernel': ['rbf', 'linear']
    }

    # Instantiate the grid search model
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='accuracy', n_jobs=-1)

    # Fit the grid search to the data
    grid_search.fit(X_train, Y_train)

    # Get the best parameters and best model
    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_

    print("Best parameters found: ", best_params)
    print("Best model found: ", best_model)

    # Evaluate the best model on the test set
    ypreds_train = best_model.predict(X_train)
    ypreds_test = best_model.predict(X_test)

    print(ypreds_test) # TODO

    # Measure accuracy
    print('Accuracy (Training):', accuracy_score(Y_train, ypreds_train))
    print('Accuracy (Test):', accuracy_score(Y_test, ypreds_test))

    # Save the model
    # Save the trained model to a file using pickle
    with open('svc_model.pkl', 'wb') as file:
        pickle.dump(best_model, file)

    with open('encoder.pkl', 'wb') as file:
        pickle.dump(encoder, file)

def predict_face(embedder, image):
    # Generating embeddings
    pred_embeddings = []
    pred_embeddings.append(get_embedding(embedder, image))
    pred_embeddings = np.asarray(pred_embeddings)

    # Prediction
    with open('svc_model.pkl', 'rb') as file:
        model = pickle.load(file)
        
    with open('encoder.pkl', 'rb') as file:
        encoder = pickle.load(file)

    result_encoded = model.predict(pred_embeddings)
    result = encoder.inverse_transform(result_encoded)

    return result[0]

def streamlit_app(faceDetection):
    embedder = FaceNet()
    st.title("Application de :blue[reconnaissance faciale]")
    st.write("Cette application fait de la reconnaissance faciale en envoyant des images sur le site. Pour l'utiliser, il suffit de télécharger une image. L'application ne reconnaît que 5 personnes. Une liste des personnalités enregistrés est disponible en bas de pages.")
    uploaded_images = st.file_uploader("Charge une image", type=['jpg', 'png', 'jpeg'], accept_multiple_files=True)
    if uploaded_images != None:
        # Read uploaded image with face_recognition
        for image in uploaded_images:
            image_vs = Image.open(image)
            image_vs = np.array(image_vs)

            name = 'Unknown'
            # Face detection
            faces = faceDetection.detectMultiScale(image_vs, scaleFactor=1.1, minNeighbors=7)

            if faces is not None:
                if len(faces) == 0:
                    st.error("Aucun visage n'a été détecté.")

            for (x, y, w, h) in faces:
                detected_face = image_vs[y:y + h, x:x + w]
                name = predict_face(embedder, detected_face)
    

                cv2.rectangle(image_vs, (x, y), (x+w, y+h), (0, 0, 255), 4)
                cv2.putText(image_vs, str(name), (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0),2,cv2.LINE_AA)

            st.info(f"Name: {name}")
            if not name == 'Visage non reconnu':
                st.image(image_vs, width=500)
    else:
        st.info("Veuillez charger une image")
            # Database
    database = {}
    parent_directory = 'Faces-Dataset/'
    for sub_dir in os.listdir(parent_directory):
        path = parent_directory + sub_dir + '/'
        for file in os.listdir(path):
            database[sub_dir] = path + file
            break

    
    st.markdown('\n')
    st.markdown('\n')
    st.markdown('\n')
    st.markdown('\n')
    st.subheader('Visages enregistrés', divider='blue')
    # Name the columns
    Name, FaceImage = st.columns(spec=[1, 1], gap='small')
    with Name:
        st.subheader('Nom & Prénom')
    with FaceImage:
        st.subheader('Images')

    # Display the registered faces (Images + Identity)
    for name, person in database.items():
        st.write('------------------------------------------------------------------------------------')
        Name, FaceImage = st.columns(spec=[1, 1], gap='small')

        with Name:
            st.write(name)

        with FaceImage:
            img = cv2.imread(person)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = np.asarray(img)
            st.image(img, width=60)


if __name__ == "__main__":
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    streamlit_app(face_cascade)


