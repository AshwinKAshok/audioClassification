# Audio Classification
This is an attempt to use deep neueal networks to classify different audio into their corresponsding audio category. 
For example, giving the input as a audio of a bird chirping, the neural network gives the output that the audio is that of a birad chirping.
Currently, this has a very bad accuracy and needs further improvement.

It has a frontend and backend component. The frontend is a simple client which can listen to audio. The backend is where the trained model is
stored and the input from the frontend is fed to the model in the backend. The output is then transferred back to the frontend client.

# TECH STACK
* Frontend: Native JavaScript and HTML/CSS
* Backend: Python Flask
