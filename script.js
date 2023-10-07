
const inputFile = document.getElementById("inputFile");
const imageElement = document.getElementById("image");
const result = document.getElementById("result");


function loadModel()
{
    const model = tf.loadGraphModel("pneumonia_detector_tfjs/model.json");
    return model;
}

inputFile.addEventListener("change", async (e) => {
    let file = e.target.files[0];
    let imageReader = new FileReader();
    imageReader.onload = async function (e) {
        imageElement.src = e.target.result;
        let pixels = tf.browser.fromPixels(imageElement).resizeBilinear([256, 256]).expandDims(0);
        result.textContent = "Đang xử lý... Xin đợi 1 lát";
        const model = await loadModel();
        pred = await model.predict(pixels).sigmoid().dataSync()[0];
        result.textContent = (pred*100).toString().slice(0, 4) + "% mắc viêm đường hô hấp"
        console.log((pred*100).toString().slice(0, 4) + "% mắc viêm đường hô hấp");
    }
    imageReader.readAsDataURL(file);
})

