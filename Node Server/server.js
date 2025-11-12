const express = require('express');
const path = require('path');
const dotenv = require('dotenv');
const axios = require('axios');
const DB = require("./config/contectDB");
const ContectSchema = require('./models/contect');

dotenv.config();

const app = express();
const PORT = process.env.PORT || 8001;

// Configuration
app.set('view engine', 'ejs');

// Middlewares
app.use(express.urlencoded({extended: true}));
app.use(express.json());
app.use(express.static(path.join(__dirname, 'public')));

app.get('/', (req, res) => {
    res.render("Home");
});

app.get('/prediction-form', (req, res) => {
    res.render("Prediction_Form");
});

app.get('/test', (req, res) => {
    res.render("Test", { data, label, slicingGC, nucleoFreq, at_gc_content, comDNA });
});

DB();

app.post("/postContect", async (req, res) => {
    let { name, email, organization, mindTopic, message } = req.body;
    await ContectSchema.create({
        name, 
        email, 
        organization, 
        mindTopic, 
        message
    });
    res.sendStatus(200);
});

const htmlString = `
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Server Error</title>
  <style>
    body {
      background-color: #f6f6f6;
      padding: 0px 20px 0px 20px;
      color: #2a2a2a;
      font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
      display: flex;
      flex-direction: column;
      justify-content: center;
      align-items: center;
      height: 100vh;
      margin: 0;
      text-align: center;
    }
    h1 {
      font-size: 3em;
      font-weight: 700;
      margin-bottom: 0.5em;
    }
    p {
      font-size: 1.2em;
      color: #747474ff;
    }
  </style>
</head>
<body>
  <h1>It's not you, it's us.</h1>
  <p>API server error (Read our documentation to enter valid data.)</p>
</body>
</html>
`;

app.post("/predict", (req, res) => {
    const data = req.body;

    axios.post("https://ankitt6174-dna-mutation-prediction.hf.space/predict", data)
        .then((responce) => {
            let prediction = responce.data;
            
            res.render('Dashboard', {
                data: prediction.topData, 
                prediction_score: prediction.Prediction, 
                comDNA: prediction.comDNA,
                slicingGC: prediction.slicingGC,
                nucleoFreq: prediction.nucleoFreq,
                at_gc_content: prediction.at_gc_content,
                mutation_name: prediction.Mutation_Label,
                DNA: prediction.DNA,
                mRNA: prediction.mRNA,
                Protein: prediction.Protein
            });
        })
        .catch((error) => {
            console.log(error);
            res.status(500).send(htmlString);

        });
});

app.listen(PORT, () => {
    console.log(`Server Running On http://localhost:${PORT}/`);
});