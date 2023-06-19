Project description
Overview of the project

The goals of this project are twofold. First, we need to identify the optimil number of categories that arxiv scientific abstracts can be grouped into. Importantly, we should not make use of the axriv category tag provided for each document, but the abstract text only. The problem of finding clusters of similar words or text is also known as topic modelling. A brief overview of the research literature suggests that the most widely used methods are based on Latent Dirichlet allocation (LDA, e.g. Blei, 2012) applied to operating on word freqquincies. The disadvantage of LDA is that there is no widely accepting method or practice for determinig the optimal number of topics/clusters and that the method id computationally intensive. I therefore decided to proceed with a simpler method - k-means clusterring applied to texts vectorized using TF-IDF and the UMAP dimensionality reduction algorithm (McInnes & Healy, 2018). The explorative analysis of a random sample of 10000 documents revealed five main clusters described by the following words and bigrams:

result, function, space, group, equation, show, graph, problem, paper, differential equation, boundary condition, lower bound, modulus space, upper bound, lie algebra, random variable, paper study, partition function, quantum group

star, galaxy, model, mass, emission, observation, source, cluster, x-ray, data, star formation, light curve, black hole, emission line, stellar population, column density, stellar mass, neutron star, power spectrum, host galaxy

model, method, data, network, system, algorithm, problem, result, paper, learning, neural network, machine learning, deep learning, result show, proposed method, paper present, paper propose, propose novel, experimental result, reinforcement learning

model, mass, theory, field, energy, result, neutrino, data, black hole, black hole, dark matter, cross section, standard model, quark mass, higgs boson, dark energy, scalar field, branching ratio, neutrino mass

quantum, state, field, model, system, phase, magnetic, result, show, energy, magnetic field, phase transition, ground state, electric field, spin-orbit coupling, low temperature, quantum state, quantum mechanic, quantum dot, quantum system

A visal representation of the most informative words per cluster:

![image](https://github.com/vankov/iris.ai.project/assets/6031570/a22ee826-d85d-4adc-96cf-8c1d4b72bc3b)


References:

Blei, D. M. (2012). Probabilistic topic models. In Communications of the ACM (Vol. 55, Issue 4, pp. 77–84). Association for Computing Machinery (ACM). https://doi.org/10.1145/2133806.2133826

Erkan, G., & Radev, D. R. (2004). LexRank: Graph-based Lexical Centrality as Salience in Text Summarization. In Journal of Artificial Intelligence Research (Vol. 22, pp. 457–479). AI Access Foundation. https://doi.org/10.1613/jair.1523

McInnes, L,& Healy, J, UMAP (2018). Uniform Manifold Approximation and Projection for Dimension Reduction. https://arxiv.org/abs/1802.03426 

Reimers, N., & Gurevych, I. (11 2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing. https://arxiv.org/abs/1908.10084





![image](https://github.com/vankov/iris.ai.project/assets/6031570/8b5af743-dced-42e8-a73d-9313b1f6313c)

Cluster #0: ['math', 'cs', 'hep-th', 'math-ph', 'cond-mat']
Cluster #1: ['astro-ph', 'physics', 'gr-qc', 'math', 'cond-mat']
Cluster #2: ['cs', 'math', 'stat', 'physics', 'eess']
Cluster #3: ['hep-ph', 'hep-th', 'astro-ph', 'gr-qc', 'hep-ex']
Cluster #4: ['cond-mat', 'physics', 'quant-ph', 'math', 'astro-ph']
