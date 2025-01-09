const express = require('express');
const multer = require('multer');
const vision = require('@google-cloud/vision');
const path = require('path');
const { spawn } = require('child_process');
require('dotenv').config();
const fs = require('fs');

const pythonPath = 'C:\\Users\\arunk\\Anaconda3\\envs\\nlp_env\\python.exe';

const app = express();
const client = new vision.ImageAnnotatorClient({
  keyFilename: process.env.GOOGLE_APPLICATION_CREDENTIALS,
});

const upload = multer({ dest: 'uploads/' });

app.use(express.static(__dirname));

app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'openingpage.html'));
});

app.post('/evaluate', upload.fields([{ name: 'image1' }, { name: 'image2' }]), async (req, res) => {
  try {
    const file1Path = req.files['image1'][0].path;
    const file2Path = req.files['image2'][0].path;

    // Extract text from both images using Google Vision API
    const [result1] = await client.textDetection(file1Path);
    const text1 = result1.textAnnotations.length > 0 ? result1.textAnnotations[0].description : 'No text detected';

    console.log(text1);
    
    const [result2] = await client.textDetection(file2Path);
    const text2 = result2.textAnnotations.length > 0 ? result2.textAnnotations[0].description : 'No text detected';

    console.log(text2);

    console.log('Using Python Path:', pythonPath);

    // Run the evaluation script
    const pythonProcess = spawn(pythonPath, [path.join(__dirname, 'miniproject2.py'), text1, text2]);
    let output = '';
    pythonProcess.stdout.on('data', (data) => {
      output += data.toString();
    });

    pythonProcess.on('close', () => {
      // Send the results back to the frontend
      const score = output.trim(); // Score from Python script
      console.log(score);
      fs.unlinkSync(file1Path);
      fs.unlinkSync(file2Path);
      res.json({ text1, text2, score });
    });
  } catch (error) {
    console.error(error);
    res.status(500).send('Error processing the images or evaluation.');
  }
});

app.listen(3000, () => {
  console.log('Server is running on port 3000');
});