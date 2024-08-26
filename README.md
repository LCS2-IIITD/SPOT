Requirements

*  pytorch == 1.8.1
*  transformers == 4.2.2
*  pytorch-pretrained-bert==0.6.2
*  sklearn == 0.21
*  scikit-learn==0.23.2
*  scipy==1.5.4
*  imblearn==0.0
*  numpy == 1.19.5
*  pandas == 1.1.5


Data CSV Format

| DialogueId | UtteranceId | Speaker | Utterance | Persona? | Persona Type |


Training and Evaluation

*  Download and place 200d GloVe word vectors in 'data/'folder
*  Prepare data in the given format and place it in 'data/' folder.
*  Execution
  + Persona Discovery
		- python dataloader.py. It prepares and dumpsthe dataframes in a format which can be used by mod-els and dataloaders to train and evaluate.
		- python model.py.  Trains  the  proposed  modelon the prepared data and dumps the best models in 'models/'
		- python getRepresent.py. Extracts the repre-sentation  for  each  instance  so  that  we  can  perform oversampling.
		- python sample.py. Use SMOTE to sample theextracted representations.
		- python model-classify.py. Train the classi-fication layers using the sampled data and dumps thebest models in 'models/'.
	+ Persona Type Identification
		- python dataloader-model2.py.py. It prepares and dumps the dataframes in a format whichcan be used by models and dataloaders to train and evaluate.
		- Set appropriate parameters in init_parameter.py
		- sh scripts/run.sh. Trains the proposed model on the prepared data and dumps the best models in 'models/'
	+ Persona Value Generation
		- python main.py. Trains the proposed model.


NOTE: We are providing the partial data. Complete data will be released subject to paper acceptance.
