// const express = require('express');
// const axios = require('axios');
// const cors = require('cors');

// const app = express();
// const port = 8000;

// // Set CORS options
// const corsOptions = {
//   origin: 'http://localhost:3000',
//   methods: 'POST',
// };

// app.use(cors(corsOptions));
// app.use(express.json());

// app.post('/predict', async (req, res) => {
//   try {
//     const flaskResponse = await axios.post('http://127.0.0.1:5000/predict', req.body);
//     res.json(flaskResponse.data);
//   } catch (error) {
//     console.error('Error making prediction:', error);
//     res.status(500).send('Error making prediction.');
//   }
// });

// app.listen(port, () => {
//   console.log(`Node.js server running on http://localhost:${port}`);
// });



const express = require('express');
const axios = require('axios');
const cors = require('cors');

const app = express();
const port = 8000;

// Set CORS options
const corsOptions = {
  origin: 'http://localhost:3000',
  methods: 'POST',
};

app.use(cors(corsOptions));
app.use(express.json());
app.post('/predictweather', async (req, res) => {
  try {
    const flaskResponse = await axios.post('http://127.0.0.1:5000/predictweather', req.body);
    res.json(flaskResponse.data);
  } catch (error) {
    console.error('Error making prediction:', error);
    res.status(500).send('Error making prediction.');
  }
});

app.post('/predictsensor', async (req, res) => {
  try {
    
    const data = req.body; // Extract the data from the request body
    console.log(data);
    console.log("abcd nodejs1");
    const flaskResponse = await axios.post('http://127.0.0.1:5000/predictsensor', data);
    console.log("abcd nodejs2");
    
    res.json(flaskResponse.data);
  } catch (error) {
    console.error('Error making prediction:', error);
    res.status(500).send('Error making prediction.');
  }
});

app.listen(port, () => {
  console.log(`Node.js server running on http://localhost:${port}`);
});
