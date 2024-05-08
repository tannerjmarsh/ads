from dataclasses import dataclass
import requests
import pandas as pd
from tqdm import tqdm
from urllib.parse import urlencode
import ast
import json
from json.decoder import JSONDecodeError
import os
import pinecone
from utils import *
import argparse
from dataset import Dataset
from langchain.vectorstores import Chroma


class Evaluation:
    """Evaluate a model on a dataset. The class is initialized with all parameters of the exeriment (eg. dataset, language model, ...). 
    
    The evaluation is done by comparing the predicted solr queries with the true solr queries. The jaccard similarity is used as the evaluation metric."""
    
    dataset: str
    datasets_dir: str | None
    model: str
    fixed_icl: bool
    n_icl: int | None

    def __init__(
            self, 
            dataset, 
            datasets_dir,
            test_path, 
            model, 
            embed_model,
            temperature,
            icl_type,
            n_icl,
            fixed_icl,
            reset_db=True,
            kwargs=None,
            vector_database_type="pinecone",
        ):
        
        self.dataset = dataset
        self.datasets_dir = datasets_dir if datasets_dir else os.path.join(os.path.dirname(__file__), 'datasets')
        self.test_path = test_path
        self.vector_database_type = vector_database_type
        self.model = model
        self.embed_model = embed_model
        self.temperature = temperature
        self.icl_type = icl_type
        self.fixed_icl = fixed_icl
        self.n_icl = n_icl
        self.reset_db = reset_db
        self.kwargs = kwargs



        # Load test and train data
        train_path = os.path.join(self.datasets_dir, f"{self.dataset}_train.csv")
        test_path = self.test_path if self.test_path else os.path.join(self.datasets_dir, f"{self.dataset}_test.csv")
        dataset = Dataset(train_path, test_path) # validate the dataset
        self.df_train = dataset.get_train()
        self.df_test = dataset.get_test()

        # TEMPORARY: do some additional processing of the datasets. This should go elsewhere
        self.df_train['solr'] = self.df_train['solr'].apply(lambda x: "q=" + x.strip())
        self.df_test['solr'] = self.df_test['solr'].apply(lambda x: x.strip())

        # Initialize pinecone database
        if not fixed_icl:
            self.__init_database(reset_db=reset_db, train_dataset=self.df_train)
        
        # Set experiment parameters of the chat model on the api service
        set_experiment_params(
            llm_model_name=self.model, 
            temperature=self.temperature, 
            embedding_model_name=self.embed_model,
            icl_type=self.icl_type,
            fixed_icl=self.fixed_icl,
            vector_database_type=self.vector_database_type
        )

    def __init_database(self, reset_db: bool =True,  train_dataset: pd.DataFrame = None):
        embedding_choices = {
            "HF": {
                "model": HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"), 
                "dim": 384},
            "OpenAI": {
                "model": OpenAIEmbeddings(),
                "dim": 1536,
            },
        }   

        vectorstore = self.__get_pinecone_langchain_client(
            "ads", 
            embedding_choices[self.embed_model]["model"], 
            embedding_choices[self.embed_model]["dim"], 
            reset=reset_db
        )

        # Embedd train examples and insert them into pinecone db if it was reset
        if reset_db:
            documents = []
            for _, row in train_dataset.iterrows():
                documents.append(
                    Document(page_content=row['nl'], metadata={'solr': row['solr']})
                )

            # print(f"documents: {len(documents)}:\n {documents}")
            print(vectorstore.add_documents(documents))

    def __get_pinecone_langchain_client(self, index_name: str, embedding, embedding_dim, reset: bool = False) -> Pinecone:
        # Initialize pinecone module
        print(f"\tInitializing pinecone...")
        pinecone.init(
        api_key=os.environ["PINECONE_API_KEY"], environment=os.environ["PINECONE_ENV"]
        )
        
        index = pinecone.Index(index_name)
        pinecone_vectorstore = Pinecone(index=index, embedding=embedding, text_key="text")

        if reset:
            print(f"\tDeleting all vectors in index...")
            index.delete(delete_all=True, namespace="")

        return pinecone_vectorstore


    def __validate_dataframe(self, df):
        """
        Ensure that the train and test dataframes have the correct columns. 
        
        They must have at least columns 'nl' and 'solr'
        """
        if not all(col in df.columns for col in ['nl', 'solr']):
            raise ValueError("Dataframe must have columns 'nl' and 'solr'")

    def run(self) -> pd.DataFrame:
        # Get true bibcodes
        print("\tGetting ground truth bibcodes...")
        self.bibcodes = batch_get_bibcodes(self.df_test['solr'])

        # Get predictions
        print("\tGetting solr predictions...")
        if self.icl_type == "rag":
            self.pred_solr = batch_nl_to_solr(
                self.df_test['nl'].tolist(), 
                model=self.model, 
                embed_model=self.embed_model, 
                temperature=self.temperature, 
                n_icl=self.n_icl, 
                fixed_icl=self.fixed_icl
            )
        elif self.icl_type == "random":
            self.pred_solr = batch_nl_to_solr(
                self.df_test['nl'].tolist(), 
                model=self.model, 
                embed_model=self.embed_model, 
                temperature=self.temperature, 
                n_icl=self.n_icl, 
                fixed_icl=self.fixed_icl,
                df_examples=self.df_train.sample(n=self.n_icl)
            )
        print("\tGetting predicted bibcodes...")
        self.pred_bibcodes = batch_get_bibcodes(self.pred_solr)

        self.df_evaluation = pd.DataFrame({
            "nl": self.df_test['nl'],
            "true_solr": self.df_test['solr'],
            "pred_solr": self.pred_solr,
            "true_bibcodes": self.bibcodes,
            "pred_bibcodes": self.pred_bibcodes, 
            "n_true_bibcodes": [len(bibcodes) for bibcodes in self.bibcodes],
            "n_pred_bibcodes": [len(bibcodes) for bibcodes in self.pred_bibcodes],
        })

        # Calculate scores
        n_intersect, n_union = [], []
        for _, row in self.df_evaluation.iterrows():
            intersection = len(set(row['true_bibcodes']).intersection(set(row['pred_bibcodes'])))
            union = len(set(row['true_bibcodes']).union(set(row['pred_bibcodes'])))

            n_intersect.append(intersection)
            n_union.append(union)
            # print(f"Overlap: {n_overlap} out of {len(row['true_bibcodes'])} papers")

        self.df_evaluation['n_intersect'] = n_intersect
        self.df_evaluation['n_union'] = n_union
        self.df_evaluation['jaccard'] = self.df_evaluation['n_intersect'] / self.df_evaluation['n_union']
    
    def print_results(self):
        print(f"Jaccard mean: {self.df_evaluation['jaccard'].mean()}")

    def save_results(self, path: str = None):
        # compile metadata and statistics
        metadata = {
            "dataset": self.dataset,
            "model": self.model,
            "embed_model": self.embed_model,
            "temperature": self.temperature,
            "n_icl": self.n_icl,
            "icl_type": self.icl_type,
            "fixed_icl": self.fixed_icl,
            "reset_db": self.reset_db,
            "test_path": self.test_path,
            "jaccard": {
                "mean": self.df_evaluation['jaccard'].mean(),
                "median": self.df_evaluation['jaccard'].median(),
                "std": self.df_evaluation['jaccard'].std(),
                "min": self.df_evaluation['jaccard'].min(),
                "max": self.df_evaluation['jaccard'].max(),
            }
        }
        # Check for overlapping keys between metadata and self.kwargs
        if self.kwargs:
            if set(metadata.keys()).intersection(set(self.kwargs.keys())):
                raise ValueError("Metadata and self.kwargs have overlapping keys.")
            # Add kwargs to metadata
            metadata.update(self.kwargs)

        if "iter" in metadata:
            name = f"{self.dataset}_{self.model}_{self.embed_model}_icl-{self.icl_type}_k{self.n_icl}_t{self.temperature}_iter{metadata['iter']}"
        else:
            name = f"{self.dataset}_{self.model}_{self.embed_model}_icl-{self.icl_type}_k{self.n_icl}_t{self.temperature}"
        # self.df_evaluation.to_csv(f"results/{name}_eval.csv", index=False)
        self.df_evaluation.to_json(f"results/{name}_eval.json", orient='records', lines=True)
        self.df_evaluation.drop(columns=["true_bibcodes", "pred_bibcodes"]).to_json(f"results/{name}_eval-nobibcodes.json", orient='records', lines=True, indent=4)



        with open(f"results/{name}_meta.json", 'w') as f:
            json.dump(metadata, f, indent=4)


class StoreDictKeyPair(argparse.Action):
     def __call__(self, parser, namespace, values, option_string=None):
         my_dict = {}
         for kv in values:
             k,v = kv.split("=")
             my_dict[k] = v
         setattr(namespace, self.dest, my_dict)

if __name__ == "__main__":

    # CLI argument parser
    parser = argparse.ArgumentParser(description="Evaluation experiment")
    parser.add_argument("dataset", type=str, help="Name of dataset to use")
    parser.add_argument("datasets_dir", type=str, help="Directory where datasets are stored")
    parser.add_argument("--test_path", type=str, default="", help="Provide a path to dataset to use instead of the default")
    parser.add_argument("--vectordb", type=str, default="pinecone", help="VDB")
    parser.add_argument("--llm", type=str, default="gpt-3.5-turbo-1106", help="LLM to use")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature to use for the LLM")
    parser.add_argument(
        '--icl_type', 
        choices=['rag', 'random', 'fixed', 'none'], 
        required=True,
        help='Determines how examples are selected for the prompt'
    )
    parser.add_argument("--n_icl", type=int, default=3, help="Number of ICL examples to use")
    parser.add_argument("--embedding", type=str, default="HF", help="Embedding model to use")
    parser.add_argument("--reset_db", action="store_true", help="Reset the database when loading train data")
    parser.add_argument("--fixed_icl", action="store_true", help="Don't take examples from the vector db. Instead insert all examples from the test dataset into the context. This overrides n_icl.")
    parser.add_argument('--kwargs', nargs='+', action=StoreDictKeyPair, help="Arbitrary keyword arguments")
    args = parser.parse_args()

    print(f"Database will be reset: {args.reset_db}")
    print(f"ICL examples are fixed: {args.fixed_icl}")

    experiment = Evaluation(
        dataset=args.dataset,
        datasets_dir=args.datasets_dir,
        test_path=args.test_path,
        model=args.llm,
        embed_model=args.embedding,
        temperature=args.temperature,
        icl_type=args.icl_type,
        n_icl=args.n_icl,
        fixed_icl=args.fixed_icl,
        reset_db=args.reset_db,
        vector_database_type=args.vectordb,
        kwargs=args.kwargs
    )

    experiment.run()
    experiment.print_results()
    experiment.save_results()

    # self.df_train['bibcodes'] = self.df_test['bibcodes'].apply(ast.literal_eval) # convert string to list
