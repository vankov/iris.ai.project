class Config:
    #Data file path
    DATA_FILE = "/data/arxiv/arxiv-metadata-oai-snapshot.json"
    #Number of documents to use for the explorative analysis
    EXPLORATIVE_SAMPLE_SIZE = 10000
    #Explorative random seed for reproducibility
    EXPLORATIVE_SAMPLE_SEED = 42
    #Number of documents to use for training the categorization model
    CATEGORIZATION_SAMPLE_SIZE = 3000
    #Categorization random seed for reproducibility
    CATEGORIZATION_SAMPLE_SEED = 22
    #Name of the Sbert model to use
    SBERT_MODEL = "sentence-transformers/bert-base-nli-mean-tokens"
    #Random seed for the clustering model
    CLUSTER_MODEL_SEED = 42
    #Processed data file path
    PROCESSED_DATA_FILE = "data.processed.json"
    #Fraction of the data (i.e. the categorization sample) to use for training and testing
    TRAIN_TEST_SPLIT = 0.8
    #Path to store the model
    MODEL_PATH = "./model"
    #Learning rate used in training
    LEARNING_RATE = 0.00001
    #Number of epochs to train
    TRAIN_EPOCHS_N = 3
    #Batch size during training
    BATCH_SIZE = 8