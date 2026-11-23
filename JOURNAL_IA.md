### Volet 2 - Utilisation de l'IA
J'ai demandé à ChatGPT comment commencer pour l'entraînement du modèle et quels sont les étapes importantes. Ensuite, je suis allé vérifier avec la documentation officielle de sckit-learn pour les différentes suggestions données par l'IA, comme ColumnTransformer, SimpleImputer ou OneHotEncoder.


### Volet 3 — Utilisation de l'IA


J'ai demandé à ChatGPT comment simplifier la saisie des coordonnées géographiques. Au départ, l'utilisateur devait entrer manuellement sa latitude et sa longitude, mais après avoir testé cette solution, nous avons trouvé qu'elle n'était pas très pratique. L'IA nous a proposé d'ajouter une option permettant d'obtenir automatiquement des coordonnées approximatives selon l'arrondissement sélectionné. Nous avons ensuite vérifié son fonctionnement dans l'interface et décidé de conserver cette modification.

Ensuite, j'ai utilisé ChatGPT pour comprendre certaines erreurs et améliorer la qualité du code. Par exemple, nous avions rencontré un problème avec la date affichée dans l'historique, qui ne se mettait pas à jour correctement lorsqu'on la modifiait. ChatGPT nous a aidés à comprendre d'où pouvait venir le problème et à trouver une solution. Nous avons ensuite vérifié le résultat dans l'application pour voir si la modification fonctionnait correctement.



Nous avons aussi utilisé ChatGPT pour améliorer l'affichage des résultats. Au départ, le graphique servait principalement à comparer les erreurs des modèles. L'IA nous a aidés à envisager une autre manière de présenter les résultats. Nous avons finalement retenu un histogramme montrant les durées historiques de résolution, avec une ligne indiquant la durée prédite. Nous avons choisi cette solution, car elle permet à l'utilisateur de mieux situer sa prédiction par rapport aux anciennes plaintes.Nous avons ensuite vérifié dans l'application que le graphique affichait correctement les durées historiques et que la ligne représentant la prédiction changeait selon les informations saisies.

Enfin, j'ai demandé à ChatGPT comment faire pour ne pas perdre la dernière prédiction lorsqu'on change de page dans l'application. Il m'a expliqué comment utiliser `st.session_state` pour conserver les résultats en mémoire. Nous avons ensuite examiné son fonctionnement dans notre code afin de comprendre comment les résultats étaient récupérés lorsqu'on revenait sur la page Résultats.


J'ai également demandé à ChatGPT pourquoi l'application pouvait se réexécuter lorsqu'on modifiait certains champs du formulaire. L'IA m'a expliqué que Streamlit relance automatiquement le script lors des interactions et m'a présenté l'utilisation de `st.form` pour regrouper les saisies. Nous avons ensuite examiné cette fonctionnalité dans notre code pour comprendre comment éviter des exécutions inutiles avant que l'utilisateur clique sur le bouton de prédiction.