# Uncertainty Adversarial Robustness

Scusate il casino, lo pulisco al più presto! :cry:

## How is this repo organized?
Il file principale è main attack, che serve per attaccare i modelli bayesiani (embedded MC dropout, injected MC dropout e Deep Ensemble) e il modello deterministico (DUQ).
Lì in pratica viene caricato il modello da attaccare con funzioni utility del modulo "models", si calcolano i risultati in caso clean, e successivamente si attacca il modello.
Sono disponibili gli attacci "MVA", "ATA" e "STAB", ma c'è da fare refactoring perché ora è fatto a metà e non è quindi più riconosciuta dal progaramma la logica degli attacchi.

Il package models comprende le classi per le varianti bayesiane di FCN (per semantic semgentation), resnet e la classe per resnet resnet DUQ;
la più importante per noi nelle fasi iniziali è la resnet, che ci serve per la classificazione.
Qui c'è una versione bayesiana della resnet classica, ed in particolare due classi, una per wrappare un ensemble ed una per il MC dropout.
In utils è possibile trovare la funzione "cuore" di tutto il package, che è "load_model";
ci permette di caricare il modello pretrainato sulla base dell'uncertainty method, della backbone e del dataset.
QUI andrà integrato poi anche il caricamento dei modelli nuovi, ovvero da RobustBench e poi dei modelli su cui faremo adv training.

Il package attacks è stato scritto quasi interamente da Daniele Angioni, e comprende la logica per gli attacchi.
Differenziamo tra "loss" (Stab, Ata, MVA, attacco a DUQ, UST) e "update" (PGS e FGSM).
Necessita un pesante refactoring, comprende purtroppo troppe cose vecchie ancora.

## Experiment Folder Subdivision
TBC
