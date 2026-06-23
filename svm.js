/**
 * SVM Solver using the Simplified SMO (Sequential Minimal Optimization) Algorithm.
 * Adheres to the standard formulation of Soft-Margin SVMs.
 */
export class SVM {
  constructor(options = {}) {
    this.kernelType = options.kernelType || 'rbf'; // 'linear', 'poly', 'rbf'
    this.C = options.C !== undefined ? options.C : 1.0; // Regularization parameter
    this.tol = options.tol || 1e-4; // Numerical tolerance
    this.maxPasses = options.maxPasses || 25; // Max epochs without alpha updates
    this.maxEpochs = options.maxEpochs || 200; // Hard limit on iterations
    
    // Kernel specific parameters
    this.rbfGamma = options.rbfGamma !== undefined ? options.rbfGamma : 1.0;
    this.polyDegree = options.polyDegree !== undefined ? options.polyDegree : 2;
    this.polyOffset = options.polyOffset !== undefined ? options.polyOffset : 1.0;
    
    // Trained model parameters
    this.data = [];
    this.labels = [];
    this.alpha = [];
    this.b = 0.0;
    this.supportVectorIndices = [];
  }

  // Kernel implementations
  kernel(x1, x2) {
    if (this.kernelType === 'linear') {
      // Linear Kernel supporting arbitrary dimensions: K(x1, x2) = x1 . x2
      let dot = 0;
      for (let i = 0; i < x1.length; i++) {
        dot += x1[i] * x2[i];
      }
      return dot;
    } else if (this.kernelType === 'poly') {
      // Polynomial Kernel: K(x1, x2) = (x1 . x2 + offset) ^ degree
      const dot = x1[0] * x2[0] + x1[1] * x2[1];
      return Math.pow(dot + this.polyOffset, this.polyDegree);
    } else if (this.kernelType === 'rbf') {
      // Radial Basis Function (RBF) Kernel: K(x1, x2) = exp(-gamma * ||x1 - x2||^2)
      const dx = x1[0] - x2[0];
      const dy = x1[1] - x2[1];
      const distSq = dx * dx + dy * dy;
      return Math.exp(-this.rbfGamma * distSq);
    }
    return 0;
  }

  /**
   * Train the Support Vector Machine using SMO.
   * @param {Array<Array<number>>} data - Array of 2D coordinates [[x, y], ...]
   * @param {Array<number>} labels - Array of class labels (+1 or -1)
   */
  train(data, labels) {
    this.data = data;
    this.labels = labels;
    const n = data.length;
    this.alpha = new Array(n).fill(0);
    this.b = 0.0;

    let passes = 0;
    let epoch = 0;

    // Cache kernel matrix to speed up training
    const kernelMatrix = [];
    for (let i = 0; i < n; i++) {
      kernelMatrix[i] = [];
      for (let j = 0; j < n; j++) {
        kernelMatrix[i][j] = this.kernel(data[i], data[j]);
      }
    }

    // Main SMO optimization loop
    while (passes < this.maxPasses && epoch < this.maxEpochs) {
      let numChangedAlphas = 0;
      
      for (let i = 0; i < n; i++) {
        // Calculate margin prediction f(x_i) = sum_j alpha_j * y_j * K(x_j, x_i) + b
        let fxi = this.b;
        for (let j = 0; j < n; j++) {
          fxi += this.alpha[j] * this.labels[j] * kernelMatrix[j][i];
        }
        
        // Error E_i = f(x_i) - y_i
        const Ei = fxi - this.labels[i];

        // Check KKT conditions: 
        // y_i * E_i < -tol and alpha_i < C OR y_i * E_i > tol and alpha_i > 0
        const yiEi = this.labels[i] * Ei;
        if ((yiEi < -this.tol && this.alpha[i] < this.C) || (yiEi > this.tol && this.alpha[i] > 0)) {
          
          // Select random helper index j different from i
          let j = Math.floor(Math.random() * (n - 1));
          if (j >= i) j++;

          // Calculate margin prediction f(x_j) and error E_j
          let fxj = this.b;
          for (let k = 0; k < n; k++) {
            fxj += this.alpha[k] * this.labels[k] * kernelMatrix[k][j];
          }
          const Ej = fxj - this.labels[j];

          // Save old alpha values
          const alphaIold = this.alpha[i];
          const alphaJold = this.alpha[j];

          // Compute L and H boundaries for clipping alpha_j
          let L = 0, H = 0;
          if (this.labels[i] !== this.labels[j]) {
            L = Math.max(0, this.alpha[j] - this.alpha[i]);
            H = Math.min(this.C, this.C + this.alpha[j] - this.alpha[i]);
          } else {
            L = Math.max(0, this.alpha[i] + this.alpha[j] - this.C);
            H = Math.min(this.C, this.alpha[i] + this.alpha[j]);
          }

          if (Math.abs(L - H) < 1e-5) continue;

          // Compute eta (similarity / curvature)
          // eta = 2 * K(xi, xj) - K(xi, xi) - K(xj, xj)
          const eta = 2 * kernelMatrix[i][j] - kernelMatrix[i][i] - kernelMatrix[j][j];
          if (eta >= 0) continue; // Must be negative for valid quadratic objective step

          // Update alpha_j
          let newAlphaJ = this.alpha[j] - (this.labels[j] * (Ei - Ej)) / eta;

          // Clip alpha_j to [L, H]
          if (newAlphaJ > H) newAlphaJ = H;
          else if (newAlphaJ < L) newAlphaJ = L;

          // Check if change is significant
          if (Math.abs(newAlphaJ - alphaJold) < 1e-5) continue;

          // Update alpha_i based on alpha_j change
          this.alpha[j] = newAlphaJ;
          this.alpha[i] = this.alpha[i] + this.labels[i] * this.labels[j] * (alphaJold - this.alpha[j]);

          // Compute b1 and b2
          const b1 = this.b - Ei - this.labels[i] * (this.alpha[i] - alphaIold) * kernelMatrix[i][i] - this.labels[j] * (this.alpha[j] - alphaJold) * kernelMatrix[i][j];
          const b2 = this.b - Ej - this.labels[i] * (this.alpha[i] - alphaIold) * kernelMatrix[i][j] - this.labels[j] * (this.alpha[j] - alphaJold) * kernelMatrix[j][j];

          // Determine bias b
          if (0 < this.alpha[i] && this.alpha[i] < this.C) {
            this.b = b1;
          } else if (0 < this.alpha[j] && this.alpha[j] < this.C) {
            this.b = b2;
          } else {
            this.b = (b1 + b2) / 2.0;
          }

          numChangedAlphas++;
        }
      }

      if (numChangedAlphas === 0) {
        passes++;
      } else {
        passes = 0;
      }
      epoch++;
    }

    // Keep track of Support Vectors indices
    this.supportVectorIndices = [];
    for (let i = 0; i < n; i++) {
      if (this.alpha[i] > 1e-5) {
        this.supportVectorIndices.push(i);
      }
    }

    return {
      epochs: epoch,
      supportVectorCount: this.supportVectorIndices.length,
      converged: passes >= this.maxPasses
    };
  }

  /**
   * Predict the margin value f(x) for a coordinate x.
   * @param {Array<number>} x - A 2D point [x, y]
   * @returns {number} - Signed distance value f(x). f(x) > 0 predicts +1, f(x) < 0 predicts -1.
   */
  predict(x) {
    let val = this.b;
    for (let i = 0; i < this.data.length; i++) {
      if (this.alpha[i] > 1e-5) {
        val += this.alpha[i] * this.labels[i] * this.kernel(this.data[i], x);
      }
    }
    return val;
  }

  /**
   * Calculate the coordinates mapping to the Z axis based on the SVM kernel decision function.
   * If a point is to be visualized in 3D:
   * Z can be mapped to f(x, y) which shows the margin distance directly as a height!
   * Alternatively, we can map to a specific kernel's raw feature map, e.g.:
   * RBF: Z = exp(-gamma * (x^2 + y^2)) (maps to a symmetric Gaussian dome)
   * Poly: Z = x^2 + y^2 (a paraboloid)
   * Linear: Z = x + y (a flat plane)
   * 
   * To show the kernel trick beautifully, we provide:
   * 1. Raw feature mapping (lifts data to a dome/bowl shape)
   * 2. SVM Decision margin mapping (lifts data according to its signed margin f(x), which makes the separating plane Z=0)
   */
  getZ(x, mappingMode = 'feature') {
    if (mappingMode === 'margin') {
      return this.predict(x);
    } else {
      // Raw feature-map projection for 3D visual geometry
      if (this.kernelType === 'rbf') {
        // Distance from center of data system [0,0]
        const dSq = x[0] * x[0] + x[1] * x[1];
        return Math.exp(-this.rbfGamma * dSq) * 3.0; // scale up for visual height
      } else if (this.kernelType === 'poly') {
        const dSq = x[0] * x[0] + x[1] * x[1];
        return dSq * 0.8; // scale up for visual height
      } else {
        // Linear: Z = w_x * x + w_y * y (we use weights from support vectors, or simple linear gradient)
        // Let's compute weights: w = sum alpha_i * y_i * x_i
        let wx = 0, wy = 0;
        for (let i = 0; i < this.data.length; i++) {
          if (this.alpha[i] > 1e-5) {
            wx += this.alpha[i] * this.labels[i] * this.data[i][0];
            wy += this.alpha[i] * this.labels[i] * this.data[i][1];
          }
        }
        return (wx * x[0] + wy * x[1]) * 0.5;
      }
    }
  }
}
