# Quantum Amplitude Damping Channel

Bachelor's thesis project in computer engineering.

Professor: Matteo Rosati

Student: Remigiusz Gabryl Della Rosa

<!-- I'm following the paper [Discriminating qubit amplitude damping channels](https://arxiv.org/pdf/2009.01000.pdf) to understand the experiments and results described to optimize the input and measurement steps to discriminate amplitude damping channels. -->

In this repository there will be the documentation and quantum experiments implemented on IBM Qiskit tool.

 ## How to get started
* follow the official documentation on site https://quantum.cloud.ibm.com/docs/en/guides/install-qiskit

* Install Python requirements

```sh
pip install -r requirements.txt
```

Should have installed the lastest version of all library required, furthermore should have installed the qadc library.
You can check it trying to run the firts cell of q_a_d notebooks

* If you want impove the experiment with real data from real backend, you must create a file .env, and it should have two global variable

```sh
IBM_TOKEN= your_token
INSTANCE= your_instance
```
  
* For future
It must refactor the all library qadc beacouse it need improve the static methods of class AmplitudeDampingConvenzioneClassica and AmplitudeDampingConvenzioneQiskit. In particular, it need to be implement more likely at the implement the interface Gate.
After this refactor, it need change the notebooks where use this classes.