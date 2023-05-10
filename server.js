

const http = require('http');
const express = require('express');
const fs = require('fs');
const torch = require("torch-js");
const csvParser = require('csv-parser');
const { MinMaxScaler } = require('machinelearn/preprocessing');
const scaler = new MinMaxScaler();
const tf = require('@tensorflow/tfjs-node');
const { createCanvas } = require('canvas');
const multer = require('multer');
const upload = multer({ dest: 'src/assets/upload' });
const D3Node = require('d3-node');


const app = express();
const port = 3000;
var photo_update = false;

app.use(express.urlencoded({ extended: true }));
app.use(express.json());
app.use(express.static('public'));

app.get('/', (req, res) => {
    res.set('Content-Type', 'text/html');
    const fp = __dirname + '/index.html';
    res.sendFile(fp);
});
app.get('/download', (req, res) => {
    const file = __dirname + '/public/esempi.zip';
    //console.log(file);
    res.download(file);
});
app.post('/upload', upload.single('datafile'), async (req, res) => {
    try {

        const selectedValue = req.body["data-type"];
        let filePathSelect;

        if (selectedValue === "apple") {
            filePathSelect = "apple.txt";
        } else if (selectedValue === "ko") {
            filePathSelect = "cocacola.txt";
        } else if (selectedValue === "ibm") {
            filePathSelect = "ibm.txt";
        } else if (selectedValue === "ge") {
            filePathSelect = "generalmotors.txt";
        } else if (selectedValue === "jpm") {
            filePathSelect = "jpmorgan.txt";
        }

        if (filePathSelect) {
            const input_from_select = __dirname + '/public/upload/' + filePathSelect;
            prediction(input_from_select);

        } else {

            // Get the uploaded file
            const file = req.file;
            console.log(file.originalname);

            // Check if the file is a text file
            if (!file.originalname.endsWith('.txt')) {
                return res.status(400).send('Invalid file type. Only text files are allowed.');
            }

            // Save the file to public/esempi/upload directory
            const fileName = file.originalname;
            const filePath = __dirname + '/public/upload/' + fileName;
            await fs.promises.rename(file.path, filePath);
            const input_to_the_model = __dirname + '/assets/upload/' + fileName;

            //res.send('File uploaded successfully.');
            console.log(input_to_the_model);

            prediction(input_to_the_model);

        }




    } catch (err) {
        console.error(err);
        res.redirect('/');
    }
});

app.get('/check_photo_saved', (req, res) => {
    // Invia se la foto è stata salvata
    if (photo_update) {
        const photo_saved = true;
        res.json({ photo_saved });
        photo_update = false;
    }
});

app.listen(port, () => {
    console.log(port);
});


async function prediction(filePath) {

    // Load the TorchScript model
    const modelPath = __dirname + '/model.pt';
    const model = new torch.ScriptModule(modelPath);

    // Read the .txt file and convert the 'Close' column to a tensor
    //const filePath = 'test.txt';
    const data = await readCSV(filePath);
    const seq_length = 59; // You can adjust this value as needed
    const input_dim = 1;
    const num_samples = data.length - seq_length;

    //Data parser
    const input_data = [];
    for (let i = data.length - seq_length; i >= 0; i--) {
        const seq = [];
        for (let j = 0; j < seq_length; j++) {
            seq.push(data[i + j].Close);
        }
        input_data.push(seq);
    }
    //console.log(input_data.length);


    const dates = data.map(row => row.Date);
    const closeValues = data.map(row => row.Close);
    //console.log(closeValues);


    // Scale the input data using MinMaxScaler
    scaler.fit(input_data);
    const scaled_input_data = scaler.transform(input_data);
    const meta = scaled_input_data.map((seq) => [seq])
    //console.log(input_data);


    // Array 3-d
    const input_data_3d = [];
    for (let i = 0; i < meta.length; i++) {
        const seq = [];
        for (let j = 0; j < meta[i][0].length; j++) {
            const val = meta[i][0][j];
            seq.push(val);
        }
        input_data_3d.push(seq);
    }
    const transposed_3d = [];
    for (let i = 0; i < input_data_3d.length; i++) {
        const slice = [];
        for (let j = 0; j < input_data_3d[i].length; j++) {
            slice.push([input_data_3d[i][j]]);
        }
        transposed_3d.push(slice);
    }

    const input_tensor = torch.tensor(transposed_3d);
    console.log("Testing Input Shape");
    console.log(input_tensor.toObject().shape);

    //Perform inference and scale back the output tensor
    const output_tensor = await model.forward(input_tensor);
    console.log("\nTesting Output Shape");
    console.log(output_tensor.toObject().shape);

    const tmp = Array.from(output_tensor.toObject().data);

    //Unscaling
    const unscaled_output_data = scaler.inverse_transform(tmp);
    unscaled_output_data.reverse();
    //console.log(unscaled_output_data);


    //const number_prediction = 5;
    //const futurePredictions = await forecast(model, closeValues, seq_length, number_prediction);
    //console.log("\nPredictions");
    //console.log(futurePredictions);


    //const predictionString = futurePredictions.join(',');
    //fs.writeFile('/predictions.txt', predictionString, (err) => {
    //    if (err) throw err;
    //});


    //Date successive per il plot
    //const lastDate = dates[dates.length - 1];
    //const nextDates = Array.from({ length: number_prediction }, (_, i) => {
    //    const date = new Date(lastDate);
    //    date.setDate(date.getDate() + i + 1);
    //    return date;
    //});

    //Plot
    generateChart(closeValues, unscaled_output_data, dates);
    console.log("Complete elaboration!");
}



async function generateChart(inputData, predictedData, dates) {
    // For the model delay 59 -> seq-length
    const delay = 59; // 57 è meglio

    // Add a delay to the predicted data
    const delayedPredictedData = Array(delay).fill(null).concat(predictedData);

    const data = [];
    for (let i = delay; i < inputData.length; i++) {
        data.push({ date: new Date(dates[i]), input: inputData[i], predicted: delayedPredictedData[i] });
    }

    const d3n = new D3Node({
        selector: '#chart',
        container: '<div id="container"><div id="chart"></div></div>'
    });

    const margin = { top: 20, right: 30, bottom: 30, left: 40 };
    const width = 800 - margin.left - margin.right;
    const height = 600 - margin.top - margin.bottom;

    const svg = d3n.createSVG(width + margin.left + margin.right, height + margin.top + margin.bottom);

    const x = d3n.d3.scaleTime()
        .domain(d3n.d3.extent(data, d => d.date))
        .range([margin.left, width - margin.right]);

    const y = d3n.d3.scaleLinear()
        .domain([d3n.d3.min(data, d => d3n.d3.min([d.input, d.predicted])), d3n.d3.max(data, d => d3n.d3.max([d.input, d.predicted]))])
        .range([height - margin.bottom, margin.top]);

    const line = d3n.d3.line()
        .defined(d => !isNaN(d.input))
        .x(d => x(d.date))
        .y(d => y(d.input));

    svg.append('path')
        .datum(data)
        .attr('fill', 'none')
        .attr('stroke', 'blue')
        .attr('stroke-width', 2.5)
        .attr('stroke-linejoin', 'round')
        .attr('d', line);



    const line2 = d3n.d3.line()
        .defined(d => !isNaN(d.predicted))
        .x(d => x(d.date))
        .y(d => y(d.predicted));

    svg.append('path')
        .datum(data)
        .attr('fill', 'none')
        .attr('stroke', 'red')
        .attr('stroke-width', 2.5)
        .attr('stroke-linejoin', 'round')
        .attr('d', line2);

    svg.append('g')
        .attr('transform', `translate(0,${height - margin.bottom})`)
        .call(d3n.d3.axisBottom(x))
        .select('.domain')
        .remove();

    svg.append('g')
        .attr('transform', `translate(${margin.left},0)`)
        .call(d3n.d3.axisLeft(y))
        .select('.domain')
        .remove();

    // Save the chart as a PNG image
    fs.writeFileSync('public/plot/chart.svg', d3n.svgString());
    console.log("Complete plot!");
    photo_update = true;
}

function readCSV(filePath) {
    return new Promise((resolve) => {
        const data = [];
        fs.createReadStream(filePath)
            .pipe(csvParser())
            .on('data', (row) => {
                data.push({
                    Date: row.Date,
                    Close: parseFloat(row.Close),
                });
            })
            .on('end', () => {
                resolve(data);
            });
    });
}