function detect(){

let text = document.getElementById("news").value;

fetch("http://127.0.0.1:5000/predict",{

method:"POST",

headers:{
"Content-Type":"application/json"
},

body:JSON.stringify({
news:text
})

})

.then(response => response.json())

.then(data => {

document.getElementById("result").innerHTML =
"Prediction: " + data.prediction +
"<br>Confidence: " + data.confidence + "%";

})

.catch(error => {
console.log(error);
});

}