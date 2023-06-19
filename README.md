**Project overview**

The goals of this project are twofold. First, we need to identify the optimil number of categories that arxiv scientific abstracts can be grouped into. Importantly, we should not make use of the axriv category tag provided for each document, but the abstract text only. The problem of finding clusters of similar words or text is also known as topic modelling. A brief overview of the research literature suggests that the most widely used methods are based on Latent Dirichlet allocation (LDA, e.g. Blei, 2012) applied to operating on word freqquincies. The disadvantage of LDA is that there is no widely accepting method or practice for determinig the optimal number of topics/clusters and that the method id computationally intensive. I therefore decided to proceed with a simpler method - k-means clusterring applied to texts vectorized using TF-IDF and the UMAP dimensionality reduction algorithm (McInnes & Healy, 2018). The explorative analysis of a random sample of 10000 documents revealed five main clusters described by the following words and bigrams:

result, function, space, group, equation, show, graph, problem, paper, differential equation, boundary condition, lower bound, modulus space, upper bound, lie algebra, random variable, paper study, partition function, quantum group

star, galaxy, model, mass, emission, observation, source, cluster, x-ray, data, star formation, light curve, black hole, emission line, stellar population, column density, stellar mass, neutron star, power spectrum, host galaxy

model, method, data, network, system, algorithm, problem, result, paper, learning, neural network, machine learning, deep learning, result show, proposed method, paper present, paper propose, propose novel, experimental result, reinforcement learning

model, mass, theory, field, energy, result, neutrino, data, black hole, black hole, dark matter, cross section, standard model, quark mass, higgs boson, dark energy, scalar field, branching ratio, neutrino mass

quantum, state, field, model, system, phase, magnetic, result, show, energy, magnetic field, phase transition, ground state, electric field, spin-orbit coupling, low temperature, quantum state, quantum mechanic, quantum dot, quantum system

A visal representation of the most informative words per cluster:

![image](https://github.com/vankov/iris.ai.project/assets/6031570/a22ee826-d85d-4adc-96cf-8c1d4b72bc3b)

Looking at the the words describing the contents of each cluster we can come with the following five categories (topics): mathematics, astrophysics, machine learning, theoretical physics/general relativity, quantum physics

More information about the solution of the clustering task can be found in the notebook ExploratoryAnalysis.ipynb

The second part of the project requires to train a classifier of research abstracts. The definition of this task doesn't make clear what data can or can not be used to accomplish it. A possible intepretation of the requirement is that we need to use to clustering model from the previous task to classify abstracts. Another way to look at the task is to use the category tags present in the arxiv data to annotate the abstarcts. I decided to the second way by selecting a number of arxiv category tags which can be used to assign a document in one of the categories identified in the previous task. To train a classifier, I first annotated the data set using the category tags present in the arxiv metedata.

During the last few years. The transformer architecture has established as the state-of-the-art approach for (almost) any natural language processing (NLP) task, including text classification. A typical way to do text classification with transformers is to apply a linear classifier to the avrage token embeddings (for example, this is what BertForSequenceClassification does in the transformers library). However this approach may not work well due to at least two reasons. First, averaging the vector representations of multiple words leads to loss of information. The problem is particularly acute when the text consists of more than a few words. Second, transformer models have a hard limit on the number of tokens they can process (e.g. Bert "base" models are limited to 512 tokens) and abstracts of research articles can be longer. To address the first problem, I suggest using a modification of the Bert model, known as SentenceBert (Reimers & Gurevych, 2019) which is specifically fine-tuned to provide adequate representations of whole sentences, rather than just words. The solution to the second problem is to summarize research abstracts into a single sentence using the LexRank algorithm (Erkan & Radev, 2014) and SBert sentence embeddings. Given a set of sentences, the LexRank algorithm selects the sentence which is most similar to all the others by applying an iterative procedure.

Due to time constraints and lack of computational resources, I trained the text classification model on random sample of just 3000 documents, 600 of which were used for validation only

The results of running the model on the validation set are displayed below:

![image](https://github.com/vankov/iris.ai.project/assets/6031570/8b5af743-dced-42e8-a73d-9313b1f6313c)

Apparently the results are far from perfect, but this is not suprising given that the the model was trained in a tiny subest of the available data. The "quantum physics" category is undersampled and no intance of it is even present in the validation set.

Source code description:

config.py - contains all the constants (e.g. filenames, sample sizes, seeds, etc) used throughout the project
process_data.py

process_data.py - prepares the data for categorization - annotates documents with the desired category and summarizes the abstracts using SBert and LexRank

train.py - trains the categorization model

api.py - a simple rest API which demonstrates how the model can be used for inference. If you have uvicorn, you can run it by typing 'uvicorn api:app' in the comamand line

**TODO:**
1. Run the clustering analysis with a larger sample
3. Explore other clustering/topic modelling algorithsm, including LDA
4. The categorization dataset is currently not balanced. This has to be addressed (e.g. by upsampling the underrepresented categories or by introducing class weights in the classifier)
5. Explore a range of values for the text classification model hyperparameters, such as learning rate, batch size and (perhaps most importantly) other SBert models.
6. It is also woth trying how the model works if the weights of the main Bert model are frozen and only the classifier is trained (i.e. do transfer learning). Given that SBert (supposedly) provides adequate representations of whole sentences, this could happen to work well.
7. Currently, the SBert model is run twice during inference - once for summarization and then again to predict the category. This can be optimized, we can get the embedding of the selected sentence from the first run and the classifier directly on it.
8. The training procedure has to be changed to the load data through a generator rather than converting it to tensorflow tensors and then loading all of them at once in memory. Also, the padding has to be done at batch level, rather than at level of the whole training set (this way the batch sequence length will be determined by the longest sequence in the batch rather than the longest sequence in the training data).
9. Predicting the test values should be done in batchs to prevent OOM if the test dataset is larger.
10. A Dockerfile to user run all the stages of the model.

**References:**

Blei, D. M. (2012). Probabilistic topic models. In Communications of the ACM (Vol. 55, Issue 4, pp. 77–84). Association for Computing Machinery (ACM). https://doi.org/10.1145/2133806.2133826

Erkan, G., & Radev, D. R. (2004). LexRank: Graph-based Lexical Centrality as Salience in Text Summarization. In Journal of Artificial Intelligence Research (Vol. 22, pp. 457–479). AI Access Foundation. https://doi.org/10.1613/jair.1523

McInnes, L,& Healy, J, UMAP (2018). Uniform Manifold Approximation and Projection for Dimension Reduction. https://arxiv.org/abs/1802.03426 

Reimers, N., & Gurevych, I. (11 2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing. https://arxiv.org/abs/1908.10084
