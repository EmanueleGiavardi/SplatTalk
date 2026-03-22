---
name: Project Subject
alwaysApply: true
---

# Project Subject

Questo è un progetto python chiamato "SplatTalk", un recente metodo che utilizza un framework basato su
Generalizable 3D Gaussian Splatting (3DGS) per produrre tokens 3D adatti ad essere "digeriti" direttamente
da un LLM pre-trainato, facendo in modo di avere un sistema in grado di comprendere la struttura 3D di una certa
scena, di avere informazioni linguistico/semantiche associate a tale struttura, e a rispondere a domande
su tale scena, il tutto utilizzando solamente posed images come input

- il codice relativo ai modelli è in `/src/model`
- il codice relativo all'encoder è in `/src/model/encoder`
- il codice relativo al decoder è in `/src/model/decoder`

