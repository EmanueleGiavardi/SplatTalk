---
name: Project Architecture
alwaysApply: true
---

# Project Subject

CONTESTO ARCHITETTURALE: SplatTalk (Generalizable 3D-Language Gaussian Splatting)

La codebase implementa una pipeline per la 3D Visual Question Answering (3D VQA) zero-shot a partire da immagini multi-vista. Si basa sul framework FreeSplat (una rete feed-forward per 3D Gaussian Splatting).

L'architettura è divisa in tre macro-componenti sequenziali:

STEP 1: Preparazione dei Dati e Autoencoder (Pseudo Ground-Truth 2D)
Questo modulo prepara il target semantico per l'addestramento 3D.

    - Estrazione Feature Visivo-Linguistiche: Nelle immagini in input, viene applicato il Vision Encoder (SigLIP) e il Multimodal Projector di LLaVA-OV. L'output per ogni immagine è una mappa di token (es. griglia 27x27 appiattita a 729 token), dove ogni token è un vettore ad alta dimensionalità in R3584 (nello spazio latente dell'LLM Qwen2).

    - Compressione (Autoencoder): Per evitare problemi di memoria GPU durante il rendering 3D, viene addestrato un Autoencoder (MLP lineare) indipendente su tutte le scene. L'Encoder comprime i vettori R3584 in vettori a bassa dimensionalità R256, normalizzati su un'ipersfera unitaria.

    - Output dello Step 1: Le Low-Dimensional Visual-Language Feature Maps a 256 dimensioni. Queste fungono da pseudo ground-truth per l'addestramento del modello 3D.

------------------------------------------------------------------------------------------------------------

STEP 2: Rete Feed-Forward 3DGS (Il Core: Encoder-Decoder)
Questa è l'architettura centrale. Invece di ottimizzare la scena da zero (per-scene optimization), usa reti neurali per predire i parametri Gaussiani in una singola passata (forward pass).
A. Gaussian Encoder (Estrazione Geometrica)

    >> Input: Immagini RGB multi-vista di una scena (es. ridimensionate a 32x32 per efficienza).

    Processo: Usa una CNN per estrarre feature multi-scala e costruisce "cost volumes" per stimare le mappe di profondità (Depth Maps).

    Retro-proiezione: Usa la profondità e le matrici della telecamera per retro-proiettare i pixel 2D nello spazio 3D, creando una point cloud iniziale di "Triplette Gaussiane".

    Fusione: Usa un modulo PTF (Pixel-wise Triplet Fusion) con una GRU per allineare e fondere triplette sovrapposte da viste diverse.

    >> Output (Triplette): Un set di tuple (μ,ω,flatente​) per ogni punto, dove:

        μ: Coordinate spaziali (x,y,z).

        ω: Peso / confidenza del punto.

        flatente​: Feature geometrico/visiva grezza compressa (NON la feature linguistica).

B. Gaussian Latent Decoder (La "Testa" Generativa)

    >> Input: Le Triplette Gaussiane (μ,ω,flatente​) fornite dall'Encoder.

    Struttura: È un Multi-Layer Perceptron (MLP) che elabora ogni tripletta in modo indipendente.

    >> Output: Sputa i parametri completi per ogni Gaussiana 3D. Oltre ai classici parametri geometrici (centro μ, scala s, rotazione Σ, opacità σ o α, colore RGB c), include una nuova testa lineare che predice il vettore semantico f ∈ R^256.

C. Joint Training e Differentiable Rendering

    Parallel Splatting: Il rasterizzatore CUDA custom esegue il rendering differenziabile proiettando le Gaussiane dalla camera virtuale. Renderizza simultaneamente l'immagine RGB (usando il colore c) e la Feature Map (usando il vettore f a 256 dim) condividendo la stessa geometria.

    Loss Function: L=∣∣I−I^∣∣2+0.05⋅LPIPS+∣∣F−F^∣∣2+1−cos(F,F^). L'errore fluisce all'indietro aggiornando i pesi dell'MLP del Decoder e della CNN dell'Encoder.

------------------------------------------------------------------------------------------------------------

STEP 3: 3D VQA Inference (Interazione con l'LLM)

Questa è la fase di query in cui il 3D-Language Gaussian Field dialoga con l'LLM (Qwen2/LLaVA).

    - Filtraggio (Entropy Adaptive Sampling): Un LLM ha un limite di contesto (es. 44 immagini). Vengono calcolate le entropie dei vettori f di tutti i milioni di gaussiane. Vengono mantenute solo le top 32.076 gaussiane con l'entropia più alta. Il vettore viene prelevato esattamente dal centro matematico (μ) della gaussiana.

    - Decompressione: I 32.076 vettori a 256 dimensioni vengono passati attraverso il Decoder dell'Autoencoder (congelato, dallo Step 1) per essere ri-proiettati nello spazio originale R3584.

    - LLM Forwarding: Questi token in R3584 diventano i "Visual Tokens". Vengono dati in pasto all'LLM (senza Positional Encoding, poiché la geometria è implicita nelle feature fritte tramite Gaussian Splatting ) insieme al prompt testuale. L'LLM genera la risposta finale.