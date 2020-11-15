# CS790-TopoAE_Classification

 The current biggest, most separable, pieces we can work on is recreating their results (A-1) and preprocessing
 our graph datasets (B-1-a). 

 Core - does a unsupervised pretraining to reconstruct graph structured data while preserving the topology of
 data within the latent space enhance the ability to classify graph structured data?

 A) Test with image data first

   1. Recreate results from Topological Autoencoders on MNIST/CIFAR10, I dont think we need to worry about spheres dataset
     a) we can use other models with/without the topology loss they used as baselines
     b) should be able to train with navigating to the correct directory and then using, follow their installation instructions for requirements.txt
     
   When running, output will be stored in the directory denoted by file_storage, please make sure this is occuring properly. Model parameters will be stored here which we will use for classification.

     python -m exp.train_model with experiments/train_model/best_runs/MNIST/TopoRegEdgeSymmetric.json device='cuda' --file_storage=BASEDIR
     
   might need to change device
   
     python -m exp.train_model with experiments/train_model/best_runs/MNIST/TopoRegEdgeSymmetric.json device='cpu' --file_storage=BASEDIR

  [ I Need Help With Training, my laptop is a toaster ]
  
 1. TopoRegEdgeSymmetric - MNIST
    
         python -m exp.train_model with experiments/train_model/best_runs/MNIST/TopoRegEdgeSymmetric.json device='cuda' --file_storage=TopoRegEdgeSymmetric_MNIST
 
 2.  LinearAE-TopoRegEdgeSymmetric - MNIST 
    
         python -m exp.train_model with experiments/train_model/best_runs/MNIST/LinearAE-TopoRegEdgeSymmetric.json device='cuda' --file_storage=LinearAE-TopoRegEdgeSymmetric_MNIST
 
 3. Vanilla - MNIST
    
         python -m exp.train_model with experiments/train_model/best_runs/MNIST/Vanilla.json device='cpu' --file_storage=Vanilla_MNIST

 4. TopoRegEdgeSymmetric - FashionMNIST
    
         python -m exp.train_model with experiments/train_model/best_runs/FashionMNIST/TopoRegEdgeSymmetric.json device='cuda' --file_storage=TopoRegEdgeSymmetric_FasionMNIST
 
 5. LinearAE-TopoRegEdgeSymmetric - FashionMNIST 
    
        python -m exp.train_model with experiments/train_model/best_runs/FashionMNIST/LinearAE-TopoRegEdgeSymmetric.json device='cuda' --file_storage=LinearAE-TopoRegEdgeSymmetric_FashionMNIST
 
 6. Vanilla - FashionMNIST
    
        python -m exp.train_model with experiments/train_model/best_runs/FashionMNIST/Vanilla.json device='cpu' --file_storage=Vanilla_FashionMNIST

 7.  TopoRegEdgeSymmetric - CIFAR10
    
         python -m exp.train_model with experiments/train_model/best_runs/CIFAR10/TopoRegEdgeSymmetric.json device='cuda' --file_storage=TopoRegEdgeSymmetric_CIFAR
 
 8.  LinearAE-TopoRegEdgeSymmetric - CIFAR10 
    
         python -m exp.train_model with experiments/train_model/best_runs/CIFAR10/LinearAE-TopoRegEdgeSymmetric.json device='cuda' --file_storage=LinearAE-TopoRegEdgeSymmetric_CIFAR10
 
9.  Vanilla - CIFAR10
    
        python -m exp.train_model with experiments/train_model/best_runs/CIFAR10/Vanilla.json device='cpu' --file_storage=Vanilla_CIFAR10
    
  
   2. Create new model with just the enocder and a classification layer after the final layer of the encoder
     a) potentially project weights from the second to last layer as well, the final layer is usually only 
        two units in their implementation, this might not be very informative for a linear classifier
     - done - added to the DeepAE and LinearAE models as classification layer 

   3. Fine tune on image classification, 
     a) compare to random initialization
     b) pretraining with normal loss

 B) Test with graph structured data

   1. Test topology preserving loss for Autoencoding:
     COLLAB, IMDB-BINARY, IMDB-MULTI, REDDIT-BINARY, REDDIT-5K, REDDIT-12k

   Datasets we (might) need to preprocess:
     i)    COLLAB 
     ii)   IMDB-BINARY
     iii)  IMDB-MULTI
     iv)   REDDIT-BINARY
     v)    REDDIT-5K REDDIT-12k

   b) Datasets preprocessed:
     i) None

   2. Fine tune on network classification
     a) compare to random initialization
     b) pretraining with normal loss

   3. We could use the same baselines for these as in "EndNote: Feature-based classification of networks"

  C) Generate figures

  D) Prepare presentation

  D) Write paper 





Extra if we finish the above with reasonable time remaining
---------------------------------------------------------------
 Consistency Regularization / Data Augmentation / Perturbation

   In "EndNote", referenced in B-3, they show that a Random Forest with a set of features that were various network statistics 
   calculated for each graph they were able to obtain SOA results in graph classification for the 5 social network datasets 
   listed above. 

   The features are below, see the supplementary information for better formatting:

   Feature label   Feature name
   NumNodes        Number of nodes
   NumEdges        Number of edges
   NumTri          Number of triangles
   ClustCoef       Global clustering coefficient
   DegAssort       Degree assortativity coefficient(Newman, 2003)
   AvgDeg          Average degree
   FracF           Fraction of nodes that are female
   FracMF          Fraction of edges that are male-female
   AvgAgeDif       Average age difference (absolute value) over edges
   FracSameZip     Fraction of edges that share the same ZIP code
   DegPC1-4        Principal components of degree distribution
   ClusPC1-4       Principal components of clustering distribution

   If these features resulted in competitive performance, maybe we can perturb some of these features and then apply some form 
   of consistency regularization. We can explore this more once we finish with the core

   Another perturbation we can try is random cropping of graphs - similar to random cropping of images, but here we just select
   some number of nodes or edges and remove them from the graph, I think we will need to do this somewhat intelligently though if we want
   this to be informative

 Apply to VAE

 Also open to other ideas if you have any, and one of my co-mentors worked on the paper mentioned above (Peter Mucha) so we 
 can talk to him if we have any questions 


