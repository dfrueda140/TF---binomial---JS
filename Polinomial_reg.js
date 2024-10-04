//Initialize the vectors to put the data of the dots
let x_vals = [];
let y_vals = [];

//Initialize the values for slope and y interception
let a,b,c; 

//optimizador de las variables. 
const learningRate = 0.4;
const optimizer = tf.train.sgd(learningRate);


function setup(){
    createCanvas(800,400);
    //inivitalize the tensors for slope and y interception. 
    a =  tf.variable(tf.scalar(random(1)));
    b =  tf.variable(tf.scalar(random(1)));
    c =  tf.variable(tf.scalar(random(1)));
}

function predict(x){
    const xs = tf.tensor1d(x);
    // Y= mx + b
    //const ys = xs.mul(m).add(b);
    // y = ax^2 + bx + c 
    const ys = xs.square().mul(a).add(xs.mul(b)).add(c)
    return ys;
}

function mousePressed(){
    // map the values in weight and height to a value between 0 and 1.
    let x = map(mouseX, 0 , width , -1 , 1);
    let y = map(mouseY, 0 , height, 1 , -1);
    // store value in the vectors.
    x_vals.push(x);
    y_vals.push(y);
}

function loss(pred,labels){
    return pred.sub(labels).square().mean();
}

function draw(){
    //condition the beguining to 
    tf.tidy(()=> {
        if(x_vals.length>0){
            const ys = tf.tensor1d(y_vals);
            optimizer.minimize(()=>loss(predict(x_vals),ys));
        }
    });

    background(0);
    stroke(255);
    strokeWeight(4);
    //Create the dots of the values store in the Xs and Ys
    for(let i = 0; i < x_vals.length; i++){
        let px =  map(x_vals[i], -1,1,0      ,width);
        let py =  map(y_vals[i], -1,1,height ,0);
        point(px,py);
    }

    const curveX = [];
    for (let x = -1 ; x < 1; x += 0.05){
        curveX.push(x);
    }
    // draw the line using the predict function to calculate Ys
    const ys = tf.tidy(()=>predict(curveX));
    let curveY = ys.dataSync(); 
    ys.dispose();
    //view the results of Y changing
    //ys.print()

    beginShape();
    noFill();
    stroke(255);
    strokeWeight(4);
    for(let i = 0; i < curveX.length ; i++){
        let x = map(curveX[i],-1,1,0,width);
        let y = map(curveY[i],-1,1,height,0);
        vertex(x,y);
    }
    endShape()
    console.log(tf.memory().numTensors);
}