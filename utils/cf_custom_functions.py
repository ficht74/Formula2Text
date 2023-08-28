import pandas as pd
import numpy as np
import json
import os
import string
import re
import inflect
import evaluate
from evaluate import load

def reshape_data_for_finetuning(data) -> list:
    """Reshape the data - every transcription get it own row

    Arguments:
        data {_type_} -- data from json-file import

    Returns:
        list -- a list with reshaped data from 800 to 4000
    """
    training_data = []
    for obj in data:
        image_name = obj["image_name"]
        formula = obj['formula']
        transcriptions = [obj['transcription1'], obj['transcription2'], obj["transcription3"], obj['transcription4'], obj['transcription5']]

        for transcription in transcriptions:
            #training_data.append([image_name +'@'+formula + '@' + transcription + '@' + transcriptions])
            training_data.append([image_name, formula, transcription,transcriptions])

    print(len(training_data))
    return training_data

#--------------------------------------------------------------------------

def convert_json_to_jsonl(json_file, jsonl_file):
    """Converts a .json file to a .jsonl file for load_dataset("imagefolder")

    Arguments:
        json_file {json} -- input data in json
        jsonl_file {jsonl} -- metadata in jsonl
    """
    # Load the JSON file
    with open(json_file, 'r') as f:
        data = json.load(f)

        # rename dict key from image_name to file_name
        for key in data:
                key['file_name'] = key.pop('image_name')

    # Write the JSONL file
    with open(jsonl_file, 'w') as f:
        for item in data:
            json.dump(item, f)
            f.write('\n')

#--------------------------------------------------------------------------

def load_test_data(test_file_json: str) -> pd.DataFrame:
    """Load test data from a json-file

    Arguments:
        test_file_json {str} -- test_data.json

    Returns:
        pd.DataFrame -- Dataframe with all test_data
    """
    try:
        df = pd.read_json(test_file_json)
        return df
    except FileNotFoundError:
        print(f"Error: File '{test_file_json}' not found.")
    except pd.errors.JSONDecodeError:
        print(f"Error: Unable to parse JSON data from file '{test_file_json}'.")
    except Exception as e:
        print(f"An error occurred while loading test data: {str(e)}")

    return pd.DataFrame()  # Return an empty DataFrame if an error occurs

#--------------------------------------------------------------------------

def load_predictions_from_file(pred_file_path):
    """
    Load a CSV file into a pandas DataFrame.

    Parameters:
        file_path (str): The path to the CSV file.

    Returns:
        pandas.DataFrame: The DataFrame containing the data from the CSV file.
    """
    try:
        dataframe = pd.read_csv(pred_file_path)
        return dataframe
    except FileNotFoundError:
        print(f"Error: The file '{pred_file_path}' was not found.")
        return None
    except Exception as e:
        print(f"Error: An error occurred while loading the file '{pred_file_path}': {e}")
        return None

#--------------------------------------------------------------------------

def calculate_bleu(data:pd.DataFrame)-> dict:
        """Calculates the BLEU of testdata

        Arguments:
            data {pd.DataFrame} -- Testdata with predictions as DataFrame

        Returns:
            dict -- of BLEU scores
        """

        predictions = []
        reference_sets = []
        bleu = evaluate.load("bleu")

        for _, row in data.iterrows():
                prediction = row['prediction'].lower()
                ref_set = row[['transcription1', 'transcription2', 'transcription3', 'transcription4', 'transcription5']].tolist()
                ref_lower = (map(lambda x: x.lower(),ref_set))
                reference_set = list(ref_lower)
                predictions.append(prediction)
                reference_sets.append(reference_set)
        
        print("# Predictions: ", len(predictions))
        print("# References : ", len(reference_sets))
        print("----------------------------------------")
                
        results = bleu.compute(predictions=predictions, references=reference_sets)
        bleu = round(results["bleu"],4)
        print(f"BLEU Score : {bleu:.4f}")
        return bleu

#--------------------------------------------------------------------------

def compute_sacrebleu(data:pd.DataFrame)-> dict:
        """Calculates the SACREBLEU of Testdata

        Arguments:
                data {pd.DataFrame} -- Testdata with predictions as DataFrame

        Returns:
                dict -- of sacrebleu values
        """
        predictions = []
        reference_sets = []
        sacrebleu = evaluate.load("sacrebleu")

        for _, row in data.iterrows():
            prediction = row['prediction'].lower()
            reference_set = row[['transcription1', 'transcription2', 'transcription3', 'transcription4', 'transcription5']].tolist()
            #ref_lower = (map(lambda x: x.lower(),ref_set))
            #reference_set = list(ref_lower)
            predictions.append(prediction)
            reference_sets.append(reference_set)
        
        print("# Predictions: ", len(predictions))
        print("# References : ", len(reference_sets))
        print("----------------------------------------")

        results = sacrebleu.compute(predictions=predictions, references=reference_sets)
        sacrebleu = round(results["score"],4)
        #['score', 'counts', 'totals', 'precisions', 'bp', 'sys_len', 'ref_len']
        print("Scarebleu  : ", sacrebleu)
        print("BP         : ", round(results["bp"],4))
        print("Counts     : ",results["counts"])
        print("Totals     : ",results["totals"])
        print("Precisions : ",results["precisions"])
        print("Sys_len    : ",results["sys_len"])
        print("Ref_len    : ",results["ref_len"])

        return results

#--------------------------------------------------------------------------

def compute_rouge(data:pd.DataFrame)-> dict:
    """Calculates ROUGE of the testdata

    Arguments:
        data {pd.DataFrame} -- Testdata with predictions as Dataframe

    Returns:
        dict -- dict of rouge values
    """

    predictions = []
    reference_sets = []
    rouge = evaluate.load("rouge")

    for _, row in data.iterrows():
        prediction = row['prediction'].lower()
        ref_set = row[['transcription1', 'transcription2', 'transcription3', 'transcription4', 'transcription5']].tolist()
        ref_lower = (map(lambda x: x.lower(),ref_set))
        reference_set = list(ref_lower)
        predictions.append(prediction)
        reference_sets.append(reference_set)
    
    print("# Predictions: ", len(predictions))
    print("# References : ", len(reference_sets))

    results = rouge.compute(predictions=predictions, references=reference_sets)

    rouge1 = results["rouge1"]
    rouge2 = results["rouge2"]
    rougeL = results["rougeL"]
    rougeLsum = results["rougeLsum"]

    print(f"Rouge1       : {rouge1:.4f}")
    print(f"Rouge2       : {rouge2:.4f}")
    print(f"RougeL       : {rougeL:.4f}")
    print(f"RougeLsum    : {rougeLsum:.4f}")
    return results

#--------------------------------------------------------------------------

def compute_wer(data:pd.DataFrame):
    """Computes the WER of all possible combinations

    Arguments:
        data {pd.DataFrame} -- Testdata with predictions as Dataframe

    Returns:
        list -- list of list of all calculates WER
        float -- average WER
    """
    predictions = []
    all_wer_scores = []
    wer = load("wer")

    for _, row in data.iterrows():
        #print("=================================================")
        prediction = row['prediction']
        wer_scores = []
        for i in range(1,6):
                reference = row[f"transcription{i}"]
                wers = wer.compute(predictions=[prediction], references=[reference])
                #print(wers)
                wer_scores.append(wers)
                #print("-----------------------------------------")
                #print("Prediction: ", prediction)
                #print("References: ", reference)
                #print("wer_scores: ", wers)
        
        all_wer_scores.append(wer_scores)
        lowest_values = [min(lst) for lst in all_wer_scores]
        avg_wer = round(sum(lowest_values) / len(lowest_values),4)

        #print("all_wer_scores: ", all_wer_scores)
    return all_wer_scores, avg_wer

#--------------------------------------------------------------------------

def generate_NLG_predictions(test_data:pd.DataFrame, model: object, tokenizer: object,prompt:str)-> pd.DataFrame:
    """Generate predictions of testdata

    Arguments:
        test_data {pd.DataFrame} -- Testdata as DataFrame
        model {object} -- Model
        tokenizer {object} -- Tokenizer
        prompt {str} -- "translate Latex to english:"

    Returns:
        pd.DataFrame -- DataFrame with Testdata and predictions
    """
    df = test_data.copy()
    y_preds = []

    for index, row in df.iterrows():
        input = prompt+row["formula"]
        encoded_input = tokenizer(input, truncation=True, return_tensors="pt")
        output = model.generate(**encoded_input, max_length=50)
        decoded_output = tokenizer.batch_decode(output, skip_special_tokens=True)
        y_preds.append(decoded_output)

    y_preds = np.array(y_preds)
    df["prediction"] = y_preds

    return df

#--------------------------------------------------------------------------

def load_test_data(test_file_json: str) -> pd.DataFrame:
    """Load test data from a json-file

    Arguments:
        test_file_json {str} -- test_data.json

    Returns:
        pd.DataFrame -- Dataframe with all test_data
    """
    try:
        df = pd.read_json(test_file_json)
        return df
    except FileNotFoundError:
        print(f"Error: File '{test_file_json}' not found.")
    except pd.errors.JSONDecodeError:
        print(f"Error: Unable to parse JSON data from file '{test_file_json}'.")
    except Exception as e:
        print(f"An error occurred while loading test data: {str(e)}")

    return pd.DataFrame()

#--------------------------------------------------------------------------

def save_evaluation_metrics_old(model_name:str, bleu_score:float, sacrebleu:float, rouge_score:dict, wer:float, filename:str):
    """Saves metrics to a metrics file

    Arguments:
        model_name {str} -- Name of the language model
        bleu_score {float} -- metric bleu_score
        sacrebleu {float} -- metric sacrebleu
        rouge_score {dict} -- metric rouge
        wer_score {float}  -- metric word error rate
        filename {str} -- name of the metrics file
    """
    
    metrics = {}
    # Check if the file exists
    if os.path.isfile(filename):
        # Load existing metrics from the file
        with open(filename, 'r') as json_file:
            metrics = json.load(json_file)

    # Adding the new entry
    metrics[model_name] = {'BLEU': bleu_score, 'ROUGE': rouge_score, 'SACREBLEU': sacrebleu, 'WER':wer}

    # Save the metrics dictionary to the file
    with open(filename, 'w') as json_file:
        json.dump(metrics, json_file)

    print(f"Entry added to {filename} successfully.")

#--------------------------------------------------------------------------

def save_evaluation_metric_wer(model_name:str, wer:list ,filename:str):
    """Saves the word error rates of all testdata to file

    Arguments:
        model_name {str} -- Name of the Language Model
        wer {list} -- List of word error rates
        filename {str} -- name of the file
    """
    metrics = {}
    # Check if the file exists
    if os.path.isfile(filename):
        # Load existing metrics from the file
        with open(filename, 'r') as json_file:
            metrics = json.load(json_file)

    # Adding the new entry
    metrics[model_name] = {'WER': wer}

    # Save the metrics dictionary to the file
    with open(filename, 'w') as json_file:
        json.dump(metrics, json_file)

    print(f"Entry added to {filename} successfully.")

#--------------------------------------------------------------------------

def compute_evaluation_metrics(df_pred:pd.DataFrame, pred_column_name:str)-> dict:

  df = df_pred.copy()
  df["references"] = df[["transcription1","transcription2","transcription3","transcription4","transcription5"]].values.tolist()
  pred_column = pred_column_name

  mul_hypo_lst = df[pred_column].tolist()
  mul_ref_lst = df["references"].tolist()
  
  bleu = evaluate.load("bleu")
  rouge = evaluate.load("rouge")
  ter = evaluate.load("ter")

  bleu_res = bleu.compute(predictions=mul_hypo_lst, references=mul_ref_lst)
  rouge_res = rouge.compute(predictions=mul_hypo_lst, references=mul_ref_lst)
  ter_res = ter.compute(predictions=mul_hypo_lst, references=mul_ref_lst)
  ter_acc = (1-(ter_res["score"]/100))*100

  bleu_score = bleu_res["bleu"]*100
  rouge1 = rouge_res["rouge1"]*100
  rouge2 = rouge_res["rouge2"]*100
  rougeL = rouge_res["rougeL"]*100
  ter_score = ter_res["score"]

  metrics ={
      "BLEU" : f"{bleu_score:,.2f}",
      "ROUGE-1" : f"{rouge1:,.2f}",
      "ROUGE-2" : f"{rouge2:,.2f}",
      "ROUGE-L" : f"{rougeL:,.2f}",
      "TER"     : f"{ter_score:,.2f}",
      "TER-ACC" : f"{ter_acc:,.2f}"
      }
  return metrics

#--------------------------------------------------------------------------

def save_evaluation_metrics(model_name:str, metric_res:dict, filename:str):    
    
    metrics = {}
    if os.path.isfile(filename):
        with open(filename, 'r') as json_file:
            metrics = json.load(json_file)

    metrics[model_name] = metric_res

    with open(filename, 'w') as json_file:
        json.dump(metrics, json_file)

    print(f"Entry added to {filename} successfully.")

#--------------------------------------------------------------------------

def post_processing_multi_predictions(df_preds:pd.DataFrame)-> pd.DataFrame:

  df = df_preds.copy()
  clean_predictions = []

  for i, row in df.iterrows():
    text = row["prediction"]
    # 1. lower case
    text_lower = text.lower()
    # 2. translate -,+ to minus or plus
    translated_text = re.sub(r' - | \+ ', translate_sign, text_lower)
    # 3. translate numbers to text
    cleaned_text = translate_numbers_to_text(translated_text)
    # 4. remove punctations
    cleaned_text = re.sub(r'[^\w\s]', ' ', cleaned_text)
    # 5. remove spaces
    clean_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    clean_predictions.append(clean_text)

  df["clean_prediction"] = clean_predictions
  return df

#--------------------------------------------------------------------------

def post_processing_single_prediction(prediction:str)-> str:

  text = prediction
  # 1. lower case
  text_lower = text.lower()
  # 2. translate -,+ to minus or plus
  translated_text = re.sub(r' - | \+ ', translate_sign, text_lower)
  # 3. translate numbers to text
  cleaned_text = translate_numbers_to_text(translated_text)
  # 4. remove punctations
  cleaned_text = re.sub(r'[^\w\s]', ' ', cleaned_text)
  # 5. remove spaces
  clean_text = re.sub(r'\s+', ' ', cleaned_text).strip()

  return clean_text

#--------------------------------------------------------------------------

def translate_sign(match):
  sign = match.group(0)
  if sign == " - ":
      return " minus "
  elif sign == " + ":
      return " plus "

#--------------------------------------------------------------------------

def translate_numbers_to_text(text):
    p = inflect.engine()
    words = text.split()
    translated_words = [p.number_to_words(word) if word.isdigit() else word for word in words]
    return ' '.join(translated_words)

#--------------------------------------------------------------------------

def find_special_characters_in_predictions(df_predictions:pd.DataFrame, column_name:str)-> list:
  """Finds all special characters in a given dataframe und column

  Arguments:
    {pd.DataFrame} -- Dataframe with a column "prediction"

  Returns:
    list -- special characters
  """
  column = column_name
  special_chars = set(string.punctuation)
  predictions = df_predictions[column].tolist()
  special_chars_in_predictions = set()

  for prediction in predictions:
      special_chars_in_prediction = set(char for char in prediction if char in special_chars)
      special_chars_in_predictions.update(special_chars_in_prediction)

  return list(special_chars_in_predictions)

#--------------------------------------------------------------------------

def compute_metrics_from_prediction_file(pred_file_path:str)-> dict:
  """computes all metrics from a given prediction data to file

    Arguments:
        pred_file_path {str} -- path to the prediction file
        
    Returns:
        dict -- with all metrics
    """
  predictions = load_predictions_from_file(pred_file_path)

  print("Start post-processing on predictions ...")
  print("---------------------------------------------------------")
  special_chars = find_special_characters_in_predictions(predictions,"prediction")
  print("Predcitions contain special characters:")
  print(special_chars)
  print("Cleaning ...")
  clean_preds = post_processing_multi_predictions(predictions)
  print("End Post-processing")
  print("---------------------------------------------------------")

  metrics_dict = compute_evaluation_metrics(clean_preds,"clean_prediction")
  print(metrics_dict)
  return metrics_dict

#--------------------------------------------------------------------------

def model_evaluation_on_testset(test_file_path:str, model:object, tokenizer:object, prompt:str) -> dict:
  """load testset and generate predictions, postprocessing, compute metrics

    Arguments:
        test_path_file {str} -- path to the testset
        model {object} -- language model
        tokenizer {object}  -- tokenizer of the model
        prompt {str} -- promt for the task for example"translate Latex to Text: "
        
    Returns:
        dict -- with all metrics
        clean_preds - with all cleaned predictions
    """
  test_data = load_test_data(test_file_path)
  preds = generate_NLG_predictions(test_data, model, tokenizer, prompt)
  clean_preds = post_processing_multi_predictions(preds)
  metrics_dict = compute_evaluation_metrics(clean_preds,"clean_prediction")
  return metrics_dict, clean_preds

#--------------------------------------------------------------------------

def compute_OCR_evaluation_metrics(df_pred:pd.DataFrame, pred_column_name:str)-> dict:

  df = df_pred.copy()
  df["references"] = df["formula"].values.tolist()
  pred_column = pred_column_name

  mul_hypo_lst = df[pred_column].tolist()
  mul_ref_lst = df["references"].tolist()

  bleu = evaluate.load("bleu")
  rouge = evaluate.load("rouge")
  ter = evaluate.load("ter")

  bleu_res = bleu.compute(predictions=mul_hypo_lst, references=mul_ref_lst)
  rouge_res = rouge.compute(predictions=mul_hypo_lst, references=mul_ref_lst)
  ter_res = ter.compute(predictions=mul_hypo_lst, references=mul_ref_lst)
  ter_acc = (1-(ter_res["score"]/100))*100

  bleu_score = bleu_res["bleu"]*100
  rouge1 = rouge_res["rouge1"]*100
  rouge2 = rouge_res["rouge2"]*100
  rougeL = rouge_res["rougeL"]*100
  ter_score = ter_res["score"]

  metrics ={
      "BLEU" : f"{bleu_score:,.2f}",
      "ROUGE-1" : f"{rouge1:,.2f}",
      "ROUGE-2" : f"{rouge2:,.2f}",
      "ROUGE-L" : f"{rougeL:,.2f}",
      "TER"     : f"{ter_score:,.2f}",
      "TER-ACC" : f"{ter_acc:,.2f}"
      }
  return metrics

#--------------------------------------------------------------------------
