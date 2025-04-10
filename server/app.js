const express = require('express');
const bodyParser = require('body-parser');
const csv = require('csv-writer').createObjectCsvWriter;
const fs = require('fs');

const app = express();
app.use(bodyParser.json());

const csvWriter = csv({
  path: 'api_data4.csv',
  header: [
    { id: 'timestamp', title: 'TIMESTAMP' },
    { id: 'filename', title: 'FILENAME' },
    { id: 'license_plate', title: 'LICENSE_PLATE' },
    { id: 'score', title: 'SCORE' }
  ],
  append: true
});

// Initialize CSV if needed
if (!fs.existsSync('api_data4.csv')) {
  csvWriter.writeRecords([]);
}

app.post('/save', async (req, res) => {
  try {
    const record = {
      timestamp: new Date().toISOString(),
      filename: req.body.filename,
      license_plate: req.body.license_plate,
      score: req.body.score
    };

    await csvWriter.writeRecords([record]);
    res.status(200).json({ 
      success: true,
      message: 'Data saved successfully'
    });
  } catch (err) {
    console.error('Server error:', err);
    res.status(500).json({
      success: false,
      message: 'Failed to save data'
    });
  }
});

app.listen(3000, () => console.log('Node.js API running on port 3000'));