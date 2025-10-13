Conda Help Instructions:

//Activate

conda activate Assignment3 
conda deactivate

//How to initialize the assignment

git clone ## cd EvolutionaryCompAssignment3

//How to update the yml for everyone to use

conda env export --from-history > environment.yml

git add environment.yml git commit -m "Update environment: add scikit-learn" git push origin main

//How to use the updated environment

conda env update --file environment.yml --prune 
conda activate Assignment3