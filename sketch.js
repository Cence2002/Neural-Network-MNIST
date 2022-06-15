class NN {
    constructor() {
    }

    create(layers) {
        this.layers = layers;
        this.len = layers.length;

        this.func = x => atan(x);
        this.funcGrad = x => 1 / (1 + x * x);

        this.build();
        this.init();
    }

    load(data) {
        this.layers = data[0];
        this.len = this.layers.length;

        this.func = data[1];
        this.funcGrad = data[2];

        this.build();
        this.init();
        this.b = data[3];
        this.w = data[4];
    }

    build() {
        this.z = new Array(this.len);
        for (let i = 0; i < this.len; i++) {
            this.z[i] = new Array(this.layers[i]);
        }

        this.a = new Array(this.len);
        this.aGrad = new Array(this.len);
        for (let i = 0; i < this.len; i++) {
            this.a[i] = new Array(this.layers[i]);
            this.aGrad[i] = new Array(this.layers[i]);
        }

        this.b = new Array(this.len);
        this.bGrad = new Array(this.len);
        this.bGradSum = new Array(this.len);
        for (let i = 0; i < this.len; i++) {
            this.b[i] = new Array(this.layers[i]);
            this.bGrad[i] = new Array(this.layers[i]);
            this.bGradSum[i] = new Array(this.layers[i]);
        }

        this.w = new Array(this.len - 1);
        this.wGrad = new Array(this.len - 1);
        this.wGradSum = new Array(this.len - 1);
        for (let i = 0; i < this.len - 1; i++) {
            this.w[i] = new Array(this.layers[i]);
            this.wGrad[i] = new Array(this.layers[i]);
            this.wGradSum[i] = new Array(this.layers[i]);
            for (let j = 0; j < this.layers[i]; j++) {
                this.w[i][j] = new Array(this.layers[i + 1]);
                this.wGrad[i][j] = new Array(this.layers[i + 1]);
                this.wGradSum[i][j] = new Array(this.layers[i + 1]);
            }
        }
    }

    init() {
        for (let i = 1; i < this.len; i++) {
            for (let j = 0; j < this.layers[i]; j++) {
                this.b[i][j] = random(-1, 1);
            }
        }

        for (let i = 0; i < this.len - 1; i++) {
            for (let j = 0; j < this.layers[i]; j++) {
                for (let k = 0; k < this.layers[i + 1]; k++) {
                    this.w[i][j][k] = random(-1, 1);
                }
            }
        }
    }


    calc(input) {
        this.a[0] = input.slice();
        for (let i = 1; i < this.len; i++) {
            for (let j = 0; j < this.layers[i]; j++) {
                this.calc_z(i, j);
                this.calc_a(i, j);
            }
        }
        return this.a[this.len - 1].slice();
    }

    calc_z(i, j) {
        let sum = this.b[i][j];
        for (let k = 0; k < this.layers[i - 1]; k++) {
            sum += this.a[i - 1][k] * this.w[i - 1][k][j];
        }
        this.z[i][j] = sum;
    }

    calc_a(i, j) {
        this.a[i][j] = this.func(this.z[i][j]);
    }


    train(batch, rate) {
        this.resetSum();
        batch.forEach(test => {
            this.addTraining(test.input, test.target);
        });
        this.descent(rate / batch.length);
    }

    resetSum() {
        for (let i = 0; i < this.len; i++) {
            this.bGradSum[i].fill(0);
        }

        for (let i = 0; i < this.len - 1; i++) {
            for (let j = 0; j < this.layers[i]; j++) {
                this.wGradSum[i][j].fill(0);
            }
        }
    }

    addTraining(input, target) {
        this.calc(input);

        if (count % (batchSize) === 0) {
            /*let list1 = new Array(this.layers[this.len - 1]);
            for (let i = 0; i < this.layers[this.len - 1]; i++) {
                list1[i] = Number(nfp(this.a[this.len - 1][i], 0, 2));
            }
            clog(list1);

            let list2 = new Array(this.layers[this.len - 1]);
            for (let i = 0; i < this.layers[this.len - 1]; i++) {
                list2[i] = Number(nfp(target[i], 0, 2));
            }
            clog(list2);*/

            clog(this.test(input, target));
        }
        count++;

        for (let i = 0; i < this.layers[this.len - 1]; i++) {
            this.aGrad[this.len - 1][i] = target[i] - this.a[this.len - 1][i];
        }

        this.calc_bGrad(2, 0);

        for (let i = this.len - 1; i > 0; i--) {
            for (let j = 0; j < this.layers[i]; j++) {
                this.calc_bGrad(i, j);
                for (let k = 0; k < this.layers[i - 1]; k++) {
                    this.calc_wGrad(i - 1, k, j);
                }
            }
            if (i > 1) {
                for (let j = 0; j < this.layers[i - 1]; j++) {
                    this.calc_aGrad(i - 1, j);
                }
            }
        }

        for (let i = 1; i < this.len; i++) {
            for (let j = 0; j < this.layers[i]; j++) {
                this.bGradSum[i][j] += this.bGrad[i][j];
            }
        }

        for (let i = 0; i < this.len - 1; i++) {
            for (let j = 0; j < this.layers[i]; j++) {
                for (let k = 0; k < this.layers[i + 1]; k++) {
                    this.wGradSum[i][j][k] += this.wGrad[i][j][k];
                }
            }
        }
    }

    calc_aGrad(i, j) {
        let sum = 0;
        for (let k = 0; k < this.layers[i + 1]; k++) {
            sum += this.bGrad[i + 1][k] * this.w[i][j][k];
        }
        this.aGrad[i][j] = sum;
    }

    calc_bGrad(i, j) {
        this.bGrad[i][j] = this.aGrad[i][j] * this.funcGrad(this.z[i][j]);
    }

    calc_wGrad(i, j, k) {
        this.wGrad[i][j][k] = this.bGrad[i + 1][k] * this.a[i][j];
    }

    descent(rate) {
        for (let i = 1; i < this.len; i++) {
            for (let j = 0; j < this.layers[i]; j++) {
                this.b[i][j] += this.bGradSum[i][j] * rate;
            }
        }

        for (let i = 0; i < this.len - 1; i++) {
            for (let j = 0; j < this.layers[i]; j++) {
                for (let k = 0; k < this.layers[i + 1]; k++) {
                    this.w[i][j][k] += this.wGradSum[i][j][k] * rate;
                }
            }
        }
    }


    test(input, target) {
        let sum = 0;
        let output = this.calc(input);
        for (let i = 0; i < this.layers[this.len - 1]; i++) {
            sum += pow(output[i] - target[i], 2);
        }
        return sum;
    }


    position(i, j) {
        return {
            x: map(i + 1, 0, this.len + 1, 0, 1),
            y: map(j + 1, 0, this.layers[i] + 1, 0, 1)
        };
    }

    show(x1, y1, x2, y2) {
        for (let i = 0; i < this.len - 1; i++) {
            for (let j = 0; j < this.layers[i]; j++) {
                for (let k = 0; k < this.layers[i + 1]; k++) {
                    if (this.w[i][j][k] >= 0) {
                        stroke(0, 1, 0);
                    } else {
                        stroke(1, 0, 0);
                    }
                    strokeWeight(sqrt(abs(this.w[i][j][k])));
                    let position1 = this.position(i, j),
                        position2 = this.position(i + 1, k);
                    line(map(position1.x, 0, 1, x1, x2), map(position1.y, 0, 1, y1, y2),
                        map(position2.x, 0, 1, x1, x2), map(position2.y, 0, 1, y1, y2));
                }
            }
        }

        stroke(1);
        strokeWeight(1);
        fill(0, 0, 1);
        for (let i = 0; i < this.len; i++) {
            for (let j = 0; j < this.layers[i]; j++) {
                let position = this.position(i, j);
                circle(map(position.x, 0, 1, x1, x2), map(position.y, 0, 1, y1, y2), 10);
            }
        }
    }

    show_a(x1, y1, x2, y2) {
        for (let i = 0; i < this.len - 1; i++) {
            for (let j = 0; j < this.layers[i]; j++) {
                for (let k = 0; k < this.layers[i + 1]; k++) {
                    if (this.w[i][j][k] >= 0) {
                        stroke(0, 1, 0);
                    } else {
                        stroke(1, 0, 0);
                    }
                    strokeWeight(sqrt(abs(this.w[i][j][k])) * this.a[i][j]);
                    let position1 = this.position(i, j),
                        position2 = this.position(i + 1, k);
                    line(map(position1.x, 0, 1, x1, x2), map(position1.y, 0, 1, y1, y2),
                        map(position2.x, 0, 1, x1, x2), map(position2.y, 0, 1, y1, y2));
                }
            }
        }

        stroke(1);
        strokeWeight(1);
        fill(0, 0, 1);
        for (let i = 0; i < this.len; i++) {
            for (let j = 0; j < this.layers[i]; j++) {
                let position = this.position(i, j);
                circle(map(position.x, 0, 1, x1, x2), map(position.y, 0, 1, y1, y2), map(this.a[i][j], -1, 1, 0, 15));
            }
        }
    }


    save() {
        let list = [];
        list.push("let loadData = [");
        list.push(JSON.stringify(this.layers) + ",");
        list.push(this.func.toString() + ",");
        list.push(this.funcGrad.toString() + ",");
        list.push(JSON.stringify(this.b) + ",");
        list.push(JSON.stringify(this.w));
        list.push("];");
        saveStrings(list, "backup_" + new Date().getTime() + ".txt");
    }
}

let start;
let count = 0;
let nn;

function setup() {
    createCanvas(800, 500);
    background(0);
    colorMode(RGB, 1);
    frameRate(1);

    nn = new NN();
    nn.load(loadData);
}


function draw() {
    background(0);
    let index = floor(random(1000));
    let test = getTest(index);
    let output = nn.calc(test.input);
    clog(output)

    nn.show_a(0, 100, 500, 400);
}

function getTest(index) {
    let input = new Array(784);
    for (let j = 0; j < 784; j++) {
        input[j] = map(data[index][j], 0, 256, -1, 1);
    }

    let target = [];
    for (let j = 0; j < 10; j++) {
        target.push(-1);
    }
    target[data[index][784]] = 1;

    return {input: input, target: target};
}

function startTimer() {
    start = new Date().getTime();
}

function endTimer() {
    clog(new Date().getTime() - start);
}

function clog(object) {
    console.log(object);
}
