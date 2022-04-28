### Automatic Humour Detection (AHD)

In this repository I present an Application Programable Interface (API) for both `REST` and `GraphQL` in Natural Language Processing(NLP) on the topic Automatic Humour Detection (AHD) on short text.

<img src="/images/cover.jpg" alt="cover" width="100%"/>

---

Project: `Automatic Humour Detection (AHD)`

Programmer: `@crispengari`

Date: `2022-04-26`

Abstract: _`Automatic Humour Detection (AHD) is a very useful topic in morden technologies. In this notebook we are going to create an Artificial Neural Network model using Deep Learning to detect humour in short texts. AHD are very useful because in model technologies such as virtual assistance and chatbots. They help Artificial Virtual Assistance and Bot to detect wether to take the conversation serious or not`._

Research Paper: [`2004.12765`](https://arxiv.org/abs/2004.12765)

Keywords: `pytorch`, `embedding`, `torchtext`, `fast-text`, `LSTM`, `RNN`, `CNN`, `tensorflow`, `keras`, `flask`, `graphql`, `rest`

Programming Language: `python`

Dataset: [`kaggle`](https://www.kaggle.com/datasets/deepcontractor/200k-short-texts-for-humor-detection)

--

### Folder structure of the server

The following is the folder structure of our `server`:

```
├───app
│   └───__pycache__
├───blueprints
│   └───__pycache__
├───exceptions
│   └───__pycache__
├───models
│   ├───pytorch
│   │   ├───static
│   │   └───__pycache__
│   ├───tensorflow
│   │   ├───static
│   │   └───__pycache__
│   └───__pycache__
└───schema
    └───__pycache_
```

### Getting started

In this section we are going to show how you can use the `ADH` server to make predictions of humour on text locally.

First you are required to have `python` installed on your computer to be more specific python version `3`

First you need to clone this repository by running the following command:

```shell
git clone https://github.com/CrispenGari/ahd-detector.git
```

And then you navigate to the server folder of this repository by running the following command:

```shell
cd ahd-detector/server
```

Next you are going to create a virtual environment `venv` by running the following command:

```shell
virtualenv venv
```

Then you need to activate the virtual environment by running the following command:

```shell
.\venv\Scripts\activate.bat
```

After activating the virtual environment you need to install the required packages by running the following command:

```shell
pip install -r requirements.txt
```

Then you are ready to start the server. To start the server you are going to run the following command:

```shell
cd api && python app.py
```

The above command will start the local server at default port of `3001` you can be able to make request to the server.

> **_Note: The tensorflow static file for the model is not available in this github repository you may need to run [this notebook](./notebooks/02_AHD_Classification.ipynb) so that you can save `.h5` model in the `./server/api/models/tensorflow/static` folder before attempting to make any request to the server._**

### Making GraphQL Request to the server

The graphql endpoint is served at the following urls:

1. http://127.0.0.1:3001/graphql
2. http://localhost:3001/graphql

This endpoint can only serve one query for detecting humour using either the `tensorflow` or `pytorch` model. If you visit the specified url's you will be represented by the `GraphiQL` interface where you can run the query as follows:

```
fragment PredictionFragment on PredictionType {
  label
  probability
  class_
  text
}

fragment ErrorFragment on ErrorType {
  field
  message
}

fragment HumourDetectionResponseFragment on PredictionResponse {
  ok
  error {
    ...ErrorFragment
  }
  prediction {
    ...PredictionFragment
  }
}

{
  predictHumour(input: {modelType: "tf", text: "What do you get if king kong sits on your piano? a flat note."}) {
    ...HumourDetectionResponseFragment
  }
}
```

If the query went well you are going to get the response in the following format:

```json
{
  "data": {
    "predictHumour": {
      "ok": true,
      "error": null,
      "prediction": {
        "label": 0,
        "probability": 1,
        "class_": "HUMOUR",
        "text": "what do you get if king kong sits on your piano? a flat note."
      }
    }
  }
}
```

### `input`

The input to the `predictHumour` takes in two arguments the:

1. `modelType`- type of graphql string it can be either `tf` or `pt` and not case sensitive

2. `text` - this is a graphql string which is the text that you want to detect if there's a humour element in it.

### Making `REST` Request to the server for Humour Detection.

You can start detecting humour from text using `rest` approach with different clients. I'm going to show few examples of how to detect humour using the following clients and api's.

### Using `Postman`

Using postman you send a `POST` request to `http://127.0.0.1:3001/api/detect-humour?model=tf` with the following request json body:

```json
{
  "text": "If the opposite of pro is con, then what is the opposite of progress"
}
```

To get the following json response:

```json
{
  "class_": "HUMOUR",
  "label": 0,
  "probability": 1.0,
  "text": "if the opposite of pro is con, then what is the opposite of progress"
}
```

> **_Note that in the request url the `query-string` `model` is required as either `tf` for tensorflow model or `pt` for pytorch model._**

### Using `cURL`

To detect humour on text using `curl` the request may look as follows:

```shell
curl -X POST http://127.0.0.1:3001/api/detect-humour?model=tf -H "Content-Type: application/json" -d "{\"text\": \"If the opposite of pro is con, then what is the opposite of progress\"}"
```

The response will be as follows:

```json
{
  "class_": "HUMOUR",
  "label": 0,
  "probability": 1.0,
  "text": "if the opposite of pro is con, then what is the opposite of progress"
}
```

### Using `javascript-fetch` API

You can use the `js` fetch api to make the request to the server locally using the following snippet code:

```js
fetch("http://127.0.0.1:3001/api/detect-humour?model=tf", {
  method: "POST",
  headers: {
    "Content-Type": "application/json",
  },
  body: JSON.stringify({
    text: "If the opposite of pro is con, then what is the opposite of progress",
  }),
})
  .then((res) => res.json())
  .then((data) => console.log(data));
```

The response will be as follows:

```json
{
  "class_": "HUMOUR",
  "label": 0,
  "probability": 1.0,
  "text": "if the opposite of pro is con, then what is the opposite of progress"
}
```

### Data

The data for training these notebooks was found on [`kaggle`](https://www.kaggle.com/datasets/deepcontractor/200k-short-texts-for-humor-detection) and you can also find the `csv` in the `data` folder.

### Notebooks

All the notebooks for training and data preparations are found in this repository.

1. [data preparation](/notebooks/00_AHD_Data_Prep.ipynb)
2. [pytorch model](/notebooks/01_AHD_Classification.ipynb)
3. [tensorflow model](/notebooks/02_AHD_Classification.ipynb)
