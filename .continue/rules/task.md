---
name: Project Task
alwaysApply: false
---

# Project Task

CONTESTO DI PROGETTO: Implementazione "Continuous Granularity" nel Gaussian Latent Decoder (SplatTalk / FreeSplat)

1. Obiettivo Principale

L'obiettivo di questo progetto è modificare l'architettura del Gaussian Latent Decoder all'interno di una pipeline basata su SplatTalk/FreeSplat. Si intende superare il "fixed-object bottleneck" implementando una logica di Continuous Granularity Segmentation ispirata a UnSAMv2.
Il Decoder non dovrà più predire un singolo vettore semantico statico per ogni Gaussiana 3D, ma dovrà generare un vettore semantico condizionato da un parametro di granularità continua g (dove g ∈ [0.1, 1.0] regola il livello di dettaglio semantico, spaziando da macro-strutture a micro-dettagli).

2. Scope e Vincoli

Area di intervento: L'intervento è circoscritto al modulo del Gaussian Latent Decoder, ovvero a quella porzione dell'architettura che prende in input le triplette gaussiane illustrate nel dettaglio dell'architettura di SplatTalk, e restituisce la vera e propria nuvola gaussiana, in cui ogni Gaussiana è definita dai 5 parametri (più il sesto relativo, appunto, alla granularità).

Componenti out-of-scope: L'estrazione delle feature 2D (SigLIP / LLaVA-OV), l'Autoencoder di compressione, il Gaussian Encoder geometrico e le logiche di rendering (CUDA rasterizer) rimangono invariati. NON si esclude che i tensori di input al Decoder e le logiche di supervisione della loss debbano essere gestiti o adattati in relazione al meccanismo di granularità da implementare 

3. Stato Attuale del Codice (Baseline)

Attualmente, il Gaussian Latent Decoder è un nn.Module (tipicamente un Multi-Layer Perceptron) che processa le "Triplette Gaussiane" generate dall'Encoder.

    Input della Baseline: Un tensore contenente i vettori latenti geometrici e visivi estratti dall'Encoder (flatente​).

    Output della Baseline: Per ogni punto 3D, predice un set di parametri per il Gaussian Splatting: coordinate spaziali (xyz), scala (scaling), rotazione (rotation), opacità (opacity), colore (rgb o armoniche sferiche) e un singolo vettore semantico statico (generalmente chiamato semantics o features, di dimensione fissa R256).

    Architettura attuale: Le feature semantiche vengono tipicamente calcolate ed emesse dall'ultimo layer lineare (nn.Linear) di un branch specifico della rete.

4. Modifiche Architetturali Previste

Per rendere il Decoder "Granularity-Aware" e raggiungere l'obiettivo del progetto, le modifiche si concentreranno sui seguenti aspetti del nn.Module:

- Aggiunta del Condizionamento (Input):

    Il metodo forward del Decoder dovrà accogliere un nuovo parametro di condizionamento: la granularità g (un tensore di valori float, tipicamente compresi tra 0.1 e 1.0).
    Questo scalare/tensore andrà in qualche modo "iniettato" nell'architettura della rete

- Modifica della "Testa" Semantica (Output):

    I branch dell'MLP responsabili della predizione della geometria (xyz, scaling, rotation, opacity) e del colore (rgb) rimarranno indipendenti da g.

    Il branch dell'MLP che predice il vettore semantics (R256) si trasformerà in una funzione condizionata. A parità di input spaziale/latente, al variare di g (es. da g=0.2 a g=0.9) la rete dovrà emettere vettori semantici differenti, corrispondenti rispettivamente al concetto macro e al micro-dettaglio.

- Backward Compatibility:

    La nuova implementazione dovrà gestire valori di default per g (ad esempio g=1.0 se non specificato in input), in modo da mantenere intatte le interfacce con i moduli chiamanti e non generare errori durante i forward pass standard o i test di validazione.