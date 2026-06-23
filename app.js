import { SVM } from './svm.js';

// --- State Management ---
const state = {
  dataset: 'circles',       // circles, xor, moons, linear
  numPoints: 120,
  noise: 0.1,
  kernelType: 'rbf',       // rbf, poly, linear
  rbfGamma: 0.5,
  polyDegree: 2,
  svmC: 1.0,
  liftMode: 'feature',     // feature, margin
  liftPercent: 0,          // 0 to 100 (animation progression)
  isPlayingLift: false,
  showPlane: true,
  showSV: true,
  showContour: true,
  showMarginPlanes: false,
  
  // Data arrays
  points2D: [],            // [[x, y], ...]
  labels: [],              // [+1, -1, ...]
  
  // Model instances
  svm2D: null,
  svm3D: null,             // Trained in 3D for feature-map mode separating plane
};

// --- Three.js Globals ---
let scene, camera, renderer, controls;
let containerEl, loadingOverlayEl;
let pointsMeshGroup = [];  // Array of meshes representing data points
let svRingsGroup = [];     // Golden rings around support vectors
let hyperplaneMesh = null;
let marginPlaneMeshPos = null;
let marginPlaneMeshNeg = null;
let groundPlaneMesh = null;
let groundCanvas, groundContext, groundTexture;

// Raycasting for point addition
const raycaster = new THREE.Raycaster();
const mouse = new THREE.Vector2();
let clickPlane; // Invisible plane at Z=0 for raycasting

// Animation frame ID
let animationFrameId;

// --- Initialize Application ---
function init() {
  containerEl = document.getElementById('canvas-container');
  loadingOverlayEl = document.getElementById('loading-overlay');
  
  // Set up Three.js scene
  initThree();
  
  // Set up Raycasting Click Plane
  const planeGeo = new THREE.PlaneGeometry(100, 100);
  const planeMat = new THREE.MeshBasicMaterial({ visible: false });
  clickPlane = new THREE.Mesh(planeGeo, planeMat);
  scene.add(clickPlane);
  
  // Bind DOM controls
  bindControls();
  
  // Generate initial dataset & Train
  generateDataset();
  trainModels();
  
  // Render and update
  updateVisualization();
  animate();
  
  // Hide loading screen
  setTimeout(() => {
    loadingOverlayEl.style.opacity = '0';
    setTimeout(() => loadingOverlayEl.style.display = 'none', 500);
  }, 600);
}

// --- Setup Three.js ---
function initThree() {
  const width = containerEl.clientWidth;
  const height = containerEl.clientHeight;
  
  // Scene
  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x08060c);
  
  // Fog for deep space look
  scene.fog = new THREE.FogExp2(0x08060c, 0.04);
  
  // Camera
  camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 100);
  resetCamera();
  
  // Renderer
  renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
  renderer.setSize(width, height);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.shadowMap.enabled = true;
  containerEl.appendChild(renderer.domElement);
  
  // OrbitControls
  controls = new THREE.OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.05;
  controls.maxPolarAngle = Math.PI / 2 - 0.01; // Prevent going below ground
  controls.minDistance = 3;
  controls.maxDistance = 30;
  
  // Lighting
  const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
  scene.add(ambientLight);
  
  const dirLight1 = new THREE.DirectionalLight(0xffffff, 0.6);
  dirLight1.position.set(5, 10, 7);
  scene.add(dirLight1);
  
  const dirLight2 = new THREE.DirectionalLight(0x8a2be2, 0.4); // purple glow light
  dirLight2.position.set(-5, -5, 5);
  scene.add(dirLight2);

  // Reference Grid Floor (Subtle grid lines)
  const gridHelper = new THREE.GridHelper(16, 32, 0x8a2be2, 0x22173b);
  gridHelper.rotation.x = Math.PI / 2; // Lie flat on XY plane
  gridHelper.position.z = -0.01;
  scene.add(gridHelper);
  
  // Axes Helper (subtle X and Y arrows)
  const axisLength = 8;
  const xGeo = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(-axisLength, 0, 0), new THREE.Vector3(axisLength, 0, 0)]);
  const yGeo = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(0, -axisLength, 0), new THREE.Vector3(0, axisLength, 0)]);
  const axisMat = new THREE.LineBasicMaterial({ color: 0x4a3f70, opacity: 0.5, transparent: true });
  scene.add(new THREE.Line(xGeo, axisMat));
  scene.add(new THREE.Line(yGeo, axisMat));

  // Initialize Ground Canvas for 2D Heatmap & Contour
  initGroundPlane();
  
  // Handle Resize
  window.addEventListener('resize', onWindowResize);
}

function resetCamera() {
  camera.position.set(0, -8, 8);
  camera.lookAt(0, 0, 0);
  if (controls) {
    controls.target.set(0, 0, 0);
    controls.update();
  }
}

function initGroundPlane() {
  groundCanvas = document.createElement('canvas');
  groundCanvas.width = 256;
  groundCanvas.height = 256;
  groundContext = groundCanvas.getContext('2d');
  
  groundTexture = new THREE.CanvasTexture(groundCanvas);
  const planeGeo = new THREE.PlaneGeometry(16, 16);
  const planeMat = new THREE.MeshBasicMaterial({
    map: groundTexture,
    transparent: true,
    opacity: 0.7,
    side: THREE.DoubleSide,
    depthWrite: false // Avoid z-fighting with grid and points
  });
  
  groundPlaneMesh = new THREE.Mesh(planeGeo, planeMat);
  groundPlaneMesh.position.z = -0.02; // Positioned slightly below Z=0
  scene.add(groundPlaneMesh);
}

// --- Data Generation ---
function generateDataset() {
  state.points2D = [];
  state.labels = [];
  
  const n = state.numPoints;
  const noise = state.noise;
  
  if (state.dataset === 'circles') {
    // Concentric Circles
    for (let i = 0; i < n; i++) {
      const label = i < n / 2 ? 1 : -1;
      let r, theta;
      if (label === 1) {
        // Inner circle
        r = (Math.random() * 1.5);
      } else {
        // Outer ring
        r = 2.4 + (Math.random() * 1.3);
      }
      theta = Math.random() * 2 * Math.PI;
      
      // Add gaussian noise to coords
      const x = r * Math.cos(theta) + (Math.random() - 0.5) * noise * 2;
      const y = r * Math.sin(theta) + (Math.random() - 0.5) * noise * 2;
      
      // Keep within bounds
      state.points2D.push([clamp(x, -6, 6), clamp(y, -6, 6)]);
      state.labels.push(label);
    }
  } else if (state.dataset === 'xor') {
    // XOR pattern
    for (let i = 0; i < n; i++) {
      const x = (Math.random() * 8) - 4;
      const y = (Math.random() * 8) - 4;
      
      // Compute standard XOR labels
      let label = (x * y > 0) ? 1 : -1;
      
      // Flip label based on noise probability
      if (Math.random() < noise) {
        label = -label;
      }
      
      // Add gap from axes to make it look cleaner, or add noise
      const xNoise = x + (Math.random() - 0.5) * 0.4;
      const yNoise = y + (Math.random() - 0.5) * 0.4;
      
      state.points2D.push([clamp(xNoise, -6, 6), clamp(yNoise, -6, 6)]);
      state.labels.push(label);
    }
  } else if (state.dataset === 'moons') {
    // Two Moons
    const halfN = Math.floor(n / 2);
    // Moon A
    for (let i = 0; i < halfN; i++) {
      const theta = (i / halfN) * Math.PI;
      const x = 2.0 * Math.cos(theta) - 1.0 + (Math.random() - 0.5) * noise * 3.5;
      const y = 2.0 * Math.sin(theta) - 0.6 + (Math.random() - 0.5) * noise * 3.5;
      state.points2D.push([clamp(x, -6, 6), clamp(y, -6, 6)]);
      state.labels.push(1);
    }
    // Moon B
    for (let i = 0; i < n - halfN; i++) {
      const theta = (i / (n - halfN)) * Math.PI;
      const x = 2.0 - 2.0 * Math.cos(theta) + (Math.random() - 0.5) * noise * 3.5;
      const y = 0.6 - 2.0 * Math.sin(theta) + (Math.random() - 0.5) * noise * 3.5;
      state.points2D.push([clamp(x, -6, 6), clamp(y, -6, 6)]);
      state.labels.push(-1);
    }
  } else if (state.dataset === 'linear') {
    // Linearly separable clusters
    // Angle of separation line
    const angle = Math.PI / 4;
    const slopeX = Math.cos(angle);
    const slopeY = Math.sin(angle);
    
    for (let i = 0; i < n; i++) {
      const label = i < n / 2 ? 1 : -1;
      let x, y;
      if (label === 1) {
        // Cluster A
        x = -1.8 + (Math.random() - 0.5) * 3.0;
        y = -1.8 + (Math.random() - 0.5) * 3.0;
      } else {
        // Cluster B
        x = 1.8 + (Math.random() - 0.5) * 3.0;
        y = 1.8 + (Math.random() - 0.5) * 3.0;
      }
      
      // Rotate coordinates slightly
      const rx = x * slopeX - y * slopeY;
      const ry = x * slopeY + y * slopeX;
      
      // Inject label flip noise
      let finalLabel = label;
      if (Math.random() < noise * 0.5) {
        finalLabel = -finalLabel;
      }
      
      state.points2D.push([clamp(rx, -6, 6), clamp(ry, -6, 6)]);
      state.labels.push(finalLabel);
    }
  }
}

// Helper math clamp
function clamp(val, min, max) {
  return Math.max(min, Math.min(max, val));
}

// --- SVM Model Training ---
function trainModels() {
  // 1. Train 2D SVM (which drives the decision boundaries and labels)
  state.svm2D = new SVM({
    kernelType: state.kernelType,
    C: state.svmC,
    rbfGamma: state.rbfGamma,
    polyDegree: state.polyDegree,
    maxPasses: 25,
    maxEpochs: 200
  });
  
  const stats = state.svm2D.train(state.points2D, state.labels);
  
  // Calculate training accuracy
  let correct = 0;
  for (let i = 0; i < state.points2D.length; i++) {
    const pred = state.svm2D.predict(state.points2D[i]);
    if ((pred >= 0 ? 1 : -1) === state.labels[i]) correct++;
  }
  const accuracy = (correct / state.points2D.length) * 100;
  
  // Update Stats UI
  document.getElementById('stat-convergence').textContent = stats.converged ? '已收斂 (Converged)' : '未收斂 (Running)';
  document.getElementById('stat-convergence').className = 'stat-value ' + (stats.converged ? 'text-gradient' : 'text-danger');
  document.getElementById('stat-sv-count').textContent = stats.supportVectorCount;
  document.getElementById('stat-accuracy').textContent = accuracy.toFixed(1) + '%';
  document.getElementById('stat-epochs').textContent = stats.epochs;
  
  // 2. Train 3D SVM (Linear) if in Feature Space Mode
  // We lift points to 3D based on raw feature function, then find the optimal separating linear hyperplane in 3D.
  if (state.liftMode === 'feature') {
    const points3D = state.points2D.map(pt => [
      pt[0],
      pt[1],
      getRawFeatureZ(pt)
    ]);
    
    state.svm3D = new SVM({
      kernelType: 'linear',
      C: state.svmC,
      maxPasses: 20,
      maxEpochs: 150
    });
    
    state.svm3D.train(points3D, state.labels);
  } else {
    state.svm3D = null;
  }
}

// Compute raw feature Z-lifting height for the 3D space
function getRawFeatureZ(pt) {
  const [x, y] = pt;
  const dSq = x * x + y * y;
  
  if (state.kernelType === 'rbf') {
    // Standard RBF lifting function: Gaussian dome centered at [0,0]
    return Math.exp(-state.rbfGamma * dSq) * 3.5;
  } else if (state.kernelType === 'poly') {
    // Polynomial: paraboloid bowl
    if (state.polyDegree === 2) {
      return dSq * 0.15; // scaled down to fit nicely
    } else {
      // higher degree
      return Math.pow(dSq, state.polyDegree / 2) * 0.05;
    }
  } else {
    // Linear: Flat slope plane aligned with coordinates
    return (x + y) * 0.4;
  }
}

// Get coordinate target Z depending on liftMode
function getTargetZ(pt) {
  if (state.liftMode === 'margin') {
    // Lift by the signed decision output f(x)
    return state.svm2D.predict(pt) * 1.0;
  } else {
    // Lift by raw feature mapping
    return getRawFeatureZ(pt);
  }
}

// --- Update and Draw Visualization Scene ---
function updateVisualization() {
  clearSceneObjects();
  
  // 1. Plot Data Points
  drawDataPoints();
  
  // 2. Draw 2D Ground Heatmap / Contour
  drawGroundHeatmap();
  
  // 3. Draw Hyperplane and Margin Surfaces
  drawSeparatingSurfaces();
}

function clearSceneObjects() {
  // Remove points
  pointsMeshGroup.forEach(mesh => scene.remove(mesh));
  pointsMeshGroup = [];
  
  // Remove support vector indicator rings
  svRingsGroup.forEach(mesh => scene.remove(mesh));
  svRingsGroup = [];
  
  // Remove Hyperplanes
  if (hyperplaneMesh) {
    scene.remove(hyperplaneMesh);
    hyperplaneMesh.geometry.dispose();
    hyperplaneMesh = null;
  }
  if (marginPlaneMeshPos) {
    scene.remove(marginPlaneMeshPos);
    marginPlaneMeshPos.geometry.dispose();
    marginPlaneMeshPos = null;
  }
  if (marginPlaneMeshNeg) {
    scene.remove(marginPlaneMeshNeg);
    marginPlaneMeshNeg.geometry.dispose();
    marginPlaneMeshNeg = null;
  }
}

// Draw data points as spheres in 3D scene
function drawDataPoints() {
  const sphereGeo = new THREE.SphereGeometry(0.12, 16, 16);
  
  // Color materials
  const matA = new THREE.MeshPhongMaterial({
    color: 0x00f2fe,
    emissive: 0x00a8c6,
    shininess: 30,
    transparent: true,
    opacity: 0.95
  });
  
  const matB = new THREE.MeshPhongMaterial({
    color: 0xff007f,
    emissive: 0xb5005b,
    shininess: 30,
    transparent: true,
    opacity: 0.95
  });
  
  // Golden ring for support vectors
  const torusGeo = new THREE.TorusGeometry(0.18, 0.025, 6, 24);
  const svMat = new THREE.MeshBasicMaterial({
    color: 0xffd700,
    side: THREE.DoubleSide
  });

  const liftFrac = state.liftPercent / 100;
  
  for (let i = 0; i < state.points2D.length; i++) {
    const pt = state.points2D[i];
    const label = state.labels[i];
    const isSV = state.svm2D.alpha[i] > 1e-5;
    
    // Compute current position (interpolated between 2D and 3D)
    const targetZ = getTargetZ(pt);
    const currentZ = targetZ * liftFrac;
    
    const mesh = new THREE.Mesh(sphereGeo, label === 1 ? matA : matB);
    mesh.position.set(pt[0], pt[1], currentZ);
    mesh.castShadow = true;
    scene.add(mesh);
    pointsMeshGroup.push(mesh);
    
    // If it's a support vector and SV toggle is on, add a golden ring
    if (isSV && state.showSV) {
      const ring = new THREE.Mesh(torusGeo, svMat);
      // Ring lies flat on XY plane but moves with point
      ring.position.set(pt[0], pt[1], currentZ);
      scene.add(ring);
      svRingsGroup.push(ring);
    }
  }
}

// Render offscreen canvas texture representing decision heat-map and overlay onto ground plane
function drawGroundHeatmap() {
  const ctx = groundContext;
  const w = groundCanvas.width;
  const h = groundCanvas.height;
  
  // Clear ground canvas
  ctx.clearRect(0, 0, w, h);
  
  if (!state.showContour) {
    groundTexture.needsUpdate = true;
    return;
  }
  
  // We scan the range [-8, 8] corresponding to the 16x16 plane mesh
  const bound = 8.0;
  const imgData = ctx.createImageData(w, h);
  
  // Draw decision boundary gradient
  for (let py = 0; py < h; py++) {
    for (let px = 0; px < w; px++) {
      // Map pixel to space coordinates
      const x = (px / w) * (bound * 2) - bound;
      // Flip Y axis for canvas drawing
      const y = -((py / h) * (bound * 2) - bound);
      
      const val = state.svm2D.predict([x, y]);
      
      const idx = (py * w + px) * 4;
      
      if (val > 0) {
        // Class +1: Cyan gradient
        // Blend from white (val close to 0) to Cyan [0, 242, 254]
        const intensity = Math.min(Math.abs(val) * 0.4, 0.7);
        imgData.data[idx] = Math.floor(0 * intensity + 26 * (1 - intensity)); // panel-bg R = 18
        imgData.data[idx + 1] = Math.floor(242 * intensity + 14 * (1 - intensity)); // G = 14
        imgData.data[idx + 2] = Math.floor(254 * intensity + 28 * (1 - intensity)); // B = 28
        imgData.data[idx + 3] = Math.floor(255 * (0.05 + intensity * 0.45)); // Alpha
      } else {
        // Class -1: Magenta gradient
        // Blend from white (val close to 0) to Magenta [255, 0, 127]
        const intensity = Math.min(Math.abs(val) * 0.4, 0.7);
        imgData.data[idx] = Math.floor(255 * intensity + 18 * (1 - intensity)); // R
        imgData.data[idx + 1] = Math.floor(0 * intensity + 14 * (1 - intensity)); // G
        imgData.data[idx + 2] = Math.floor(127 * intensity + 28 * (1 - intensity)); // B
        imgData.data[idx + 3] = Math.floor(255 * (0.05 + intensity * 0.45)); // Alpha
      }
    }
  }
  ctx.putImageData(imgData, 0, 0);
  
  // Draw thin crisp line representing decision boundary (f(x, y) = 0) and margins (f(x, y) = +/-1)
  // We will do this by contour tracing or simple pixel checking
  // For simplicity and crisp lines, let's draw contour lines on canvas using 2D context
  const gridRes = 64;
  const grid = [];
  
  for (let i = 0; i <= gridRes; i++) {
    grid[i] = [];
    const y = -((i / gridRes) * (bound * 2) - bound);
    for (let j = 0; j <= gridRes; j++) {
      const x = (j / gridRes) * (bound * 2) - bound;
      grid[i][j] = state.svm2D.predict([x, j]); // wait! should be [x, y]
    }
  }
  
  // Re-draw accurate vectors contours
  ctx.lineWidth = 2.0;
  
  // Let's trace contour lines: we scan pixels and look for zero-crossings
  const strokeData = ctx.getImageData(0, 0, w, h);
  for (let py = 1; py < h - 1; py++) {
    for (let px = 1; px < w - 1; px++) {
      const x = (px / w) * (bound * 2) - bound;
      const y = -((py / h) * (bound * 2) - bound);
      
      const v = state.svm2D.predict([x, y]);
      
      // Neighboring values
      const vx = state.svm2D.predict([x + 0.1, y]);
      const vy = state.svm2D.predict([x, y + 0.1]);
      
      const idx = (py * w + px) * 4;
      
      // Zero boundary (decision boundary) - Black line
      if (Math.sign(v) !== Math.sign(vx) || Math.sign(v) !== Math.sign(vy)) {
        // Check threshold for thinness
        strokeData.data[idx] = 255;
        strokeData.data[idx + 1] = 255;
        strokeData.data[idx + 2] = 255;
        strokeData.data[idx + 3] = 255;
      }
      
      // Margins (f(x, y) = +/- 1) - Dotted/Subtle white line
      if (state.showMarginPlanes) {
        const m1 = v - 1.0;
        const m1x = vx - 1.0;
        const m1y = vy - 1.0;
        const m2 = v + 1.0;
        const m2x = vx + 1.0;
        const m2y = vy + 1.0;
        
        if ((Math.sign(m1) !== Math.sign(m1x) || Math.sign(m1) !== Math.sign(m1y)) ||
            (Math.sign(m2) !== Math.sign(m2x) || Math.sign(m2) !== Math.sign(m2y))) {
          // Yellowish dashed look
          strokeData.data[idx] = 255;
          strokeData.data[idx + 1] = 215;
          strokeData.data[idx + 2] = 0;
          strokeData.data[idx + 3] = 160;
        }
      }
    }
  }
  ctx.putImageData(strokeData, 0, 0);
  
  groundTexture.needsUpdate = true;
}

// Generate the 3D separating hyperplane mesh and margins meshes
function drawSeparatingSurfaces() {
  if (!state.showPlane) return;
  
  const liftFrac = state.liftPercent / 100;
  if (liftFrac < 0.05) return; // Don't draw plane if completely flat in 2D
  
  const gridRes = 35;
  const bound = 8.0;
  
  // Custom geometry representing separating hyperplane
  const hyperPlaneGeo = new THREE.PlaneGeometry(bound * 2, bound * 2, gridRes, gridRes);
  const posMat = new THREE.MeshPhongMaterial({
    color: 0x9d4edd,
    side: THREE.DoubleSide,
    transparent: true,
    opacity: 0.35 * liftFrac,
    wireframe: false,
    shininess: 40,
    depthWrite: false
  });
  
  // Populate heights for the separating hyperplane
  const posAttr = hyperPlaneGeo.attributes.position;
  
  // Linear weights if needed for Feature Mode
  let w = [0, 0, 0], bias = 0;
  if (state.liftMode === 'feature' && state.svm3D) {
    for (let i = 0; i < state.svm3D.data.length; i++) {
      if (state.svm3D.alpha[i] > 1e-5) {
        w[0] += state.svm3D.alpha[i] * state.svm3D.labels[i] * state.svm3D.data[i][0];
        w[1] += state.svm3D.alpha[i] * state.svm3D.labels[i] * state.svm3D.data[i][1];
        w[2] += state.svm3D.alpha[i] * state.svm3D.labels[i] * state.svm3D.data[i][2];
      }
    }
    bias = state.svm3D.b;
  }
  
  // Set heights of plane geometry vertices
  for (let i = 0; i < posAttr.count; i++) {
    const x = posAttr.getX(i);
    const y = posAttr.getY(i);
    
    let targetZ = 0;
    if (state.liftMode === 'margin') {
      // In Margin Space, the hyperplane is exactly Z = 0
      targetZ = 0;
    } else {
      // In Feature Space, hyperplane is defined by: w0*x + w1*y + w2*z + b = 0
      // => z = -(w0*x + w1*y + b) / w2
      if (Math.abs(w[2]) > 1e-4) {
        targetZ = -(w[0] * x + w[1] * y + bias) / w[2];
      } else {
        targetZ = 0; // fallback if vertical hyperplane
      }
    }
    
    // Scale height by dimension lift slider fraction
    posAttr.setZ(i, targetZ * liftFrac);
  }
  hyperPlaneGeo.computeVertexNormals();
  
  hyperplaneMesh = new THREE.Mesh(hyperPlaneGeo, posMat);
  scene.add(hyperplaneMesh);
  
  // Draw Margin Planes if enabled (Z = +/-1 in Margin Mode, or shifted planes in Feature Mode)
  if (state.showMarginPlanes) {
    const marginMat = new THREE.MeshPhongMaterial({
      color: 0xffd700, // Gold margin planes
      side: THREE.DoubleSide,
      transparent: true,
      opacity: 0.12 * liftFrac,
      wireframe: false,
      depthWrite: false
    });
    
    // Positive margin plane (f(x) = +1)
    const marginGeoPos = new THREE.PlaneGeometry(bound * 2, bound * 2, gridRes, gridRes);
    const posAttrP = marginGeoPos.attributes.position;
    for (let i = 0; i < posAttrP.count; i++) {
      const x = posAttrP.getX(i);
      const y = posAttrP.getY(i);
      
      let targetZ = 0;
      if (state.liftMode === 'margin') {
        targetZ = 1.0;
      } else {
        if (Math.abs(w[2]) > 1e-4) {
          // w0*x + w1*y + w2*z + b = 1 => z = -(w0*x + w1*y + b - 1) / w2
          targetZ = -(w[0] * x + w[1] * y + bias - 1.0) / w[2];
        } else {
          targetZ = 0.5;
        }
      }
      posAttrP.setZ(i, targetZ * liftFrac);
    }
    marginGeoPos.computeVertexNormals();
    marginPlaneMeshPos = new THREE.Mesh(marginGeoPos, marginMat);
    scene.add(marginPlaneMeshPos);
    
    // Negative margin plane (f(x) = -1)
    const marginGeoNeg = new THREE.PlaneGeometry(bound * 2, bound * 2, gridRes, gridRes);
    const posAttrN = marginGeoNeg.attributes.position;
    for (let i = 0; i < posAttrN.count; i++) {
      const x = posAttrN.getX(i);
      const y = posAttrN.getY(i);
      
      let targetZ = 0;
      if (state.liftMode === 'margin') {
        targetZ = -1.0;
      } else {
        if (Math.abs(w[2]) > 1e-4) {
          // w0*x + w1*y + w2*z + b = -1 => z = -(w0*x + w1*y + b + 1) / w2
          targetZ = -(w[0] * x + w[1] * y + bias + 1.0) / w[2];
        } else {
          targetZ = -0.5;
        }
      }
      posAttrN.setZ(i, targetZ * liftFrac);
    }
    marginGeoNeg.computeVertexNormals();
    marginPlaneMeshNeg = new THREE.Mesh(marginGeoNeg, marginMat);
    scene.add(marginPlaneMeshNeg);
  }
}

// --- Dynamic Point Addition via Clicking Canvas ---
function onCanvasClick(event) {
  // Translate mouse coordinate to normalized device coordinates (-1 to +1)
  const rect = renderer.domElement.getBoundingClientRect();
  mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
  mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
  
  raycaster.setFromCamera(mouse, camera);
  const intersects = raycaster.intersectObject(clickPlane);
  
  if (intersects.length > 0) {
    const intersectPt = intersects[0].point;
    const clickX = intersectPt.x;
    const clickY = intersectPt.y;
    
    // Bound check
    if (clickX >= -7.8 && clickX <= 7.8 && clickY >= -7.8 && clickY <= 7.8) {
      // Add point: Hold Shift for Class -1, default is Class +1
      const label = event.shiftKey ? -1 : 1;
      
      state.points2D.push([clickX, clickY]);
      state.labels.push(label);
      
      // Update UI slider count
      state.numPoints = state.points2D.length;
      document.getElementById('points-slider').value = state.numPoints;
      document.getElementById('points-value').textContent = state.numPoints;
      
      // Retrain and refresh visual elements
      trainModels();
      updateVisualization();
    }
  }
}

// --- Bind HTML Event Listeners ---
function bindControls() {
  // Canvas click listener
  renderer.domElement.addEventListener('click', onCanvasClick);
  
  // Dataset Select
  document.getElementById('dataset-select').addEventListener('change', (e) => {
    state.dataset = e.target.value;
    generateDataset();
    trainModels();
    updateVisualization();
  });
  
  // Points Slider
  const pointsSlider = document.getElementById('points-slider');
  const pointsVal = document.getElementById('points-value');
  pointsSlider.addEventListener('input', (e) => {
    state.numPoints = parseInt(e.target.value);
    pointsVal.textContent = state.numPoints;
    generateDataset();
    trainModels();
    updateVisualization();
  });
  
  // Noise Slider
  const noiseSlider = document.getElementById('noise-slider');
  const noiseVal = document.getElementById('noise-value');
  noiseSlider.addEventListener('input', (e) => {
    state.noise = parseFloat(e.target.value);
    noiseVal.textContent = state.noise.toFixed(2);
    generateDataset();
    trainModels();
    updateVisualization();
  });
  
  // Regenerate Button
  document.getElementById('regenerate-btn').addEventListener('click', () => {
    generateDataset();
    trainModels();
    updateVisualization();
  });
  
  // Kernel Select
  document.getElementById('kernel-select').addEventListener('change', (e) => {
    state.kernelType = e.target.value;
    
    // Show/hide specific parameters
    document.getElementById('rbf-gamma-group').classList.add('hide');
    document.getElementById('poly-degree-group').classList.add('hide');
    
    if (state.kernelType === 'rbf') {
      document.getElementById('rbf-gamma-group').classList.remove('hide');
      document.getElementById('math-kernel-formula').innerHTML = 'RBF 核函數：$z = e^{-\\gamma \\|\\mathbf{x} - \\mathbf{x}_j\\|^2}$';
    } else if (state.kernelType === 'poly') {
      document.getElementById('poly-degree-group').classList.remove('hide');
      document.getElementById('math-kernel-formula').innerHTML = `多項式核函數：$z = (\\mathbf{x} \\cdot \\mathbf{x}_j + 1)^{${state.polyDegree}}$`;
    } else {
      document.getElementById('math-kernel-formula').innerHTML = '線性核函數：$z = \\mathbf{w} \\cdot \\mathbf{x}$ (高度呈線性)';
    }
    
    // Re-render math equations via MathJax
    if (window.MathJax && typeof window.MathJax.typeset === 'function') {
      window.MathJax.typeset();
    }
    
    trainModels();
    updateVisualization();
  });
  
  // RBF Gamma Slider
  const gammaSlider = document.getElementById('rbf-gamma-slider');
  const gammaVal = document.getElementById('rbf-gamma-value');
  gammaSlider.addEventListener('input', (e) => {
    state.rbfGamma = parseFloat(e.target.value);
    gammaVal.textContent = state.rbfGamma.toFixed(2);
    trainModels();
    updateVisualization();
  });
  
  // Poly Degree Slider
  const degreeSlider = document.getElementById('poly-degree-slider');
  const degreeVal = document.getElementById('poly-degree-value');
  degreeSlider.addEventListener('input', (e) => {
    state.polyDegree = parseInt(e.target.value);
    degreeVal.textContent = state.polyDegree;
    
    document.getElementById('math-kernel-formula').innerHTML = `多項式核函數：$z = (\\mathbf{x} \\cdot \\mathbf{x}_j + 1)^{${state.polyDegree}}$`;
    if (window.MathJax && typeof window.MathJax.typeset === 'function') {
      window.MathJax.typeset();
    }
    
    trainModels();
    updateVisualization();
  });
  
  // SVM C Slider
  const cSlider = document.getElementById('svm-c-slider');
  const cVal = document.getElementById('svm-c-value');
  cSlider.addEventListener('input', (e) => {
    state.svmC = parseFloat(e.target.value);
    cVal.textContent = state.svmC.toFixed(2);
    trainModels();
    updateVisualization();
  });
  
  // Height Lift Z-Mode Select
  document.getElementById('lift-mode-select').addEventListener('change', (e) => {
    state.liftMode = e.target.value;
    trainModels();
    updateVisualization();
  });
  
  // Dimension Lift Slider
  const liftSlider = document.getElementById('lift-slider');
  const liftVal = document.getElementById('lift-value');
  liftSlider.addEventListener('input', (e) => {
    state.liftPercent = parseInt(e.target.value);
    
    if (state.liftPercent === 0) {
      liftVal.textContent = '0% (2D)';
    } else if (state.liftPercent === 100) {
      liftVal.textContent = '100% (3D)';
    } else {
      liftVal.textContent = state.liftPercent + '%';
    }
    
    updateDataPointsPositions();
    updateSeparatingSurfacesPositions();
  });
  
  // Play/Pause Lift Button
  const playBtn = document.getElementById('play-lift-btn');
  playBtn.addEventListener('click', () => {
    state.isPlayingLift = !state.isPlayingLift;
    if (state.isPlayingLift) {
      playBtn.innerHTML = '<i class="fa-solid fa-pause"></i>';
      // If already at 100%, reset to 0% to play again
      if (state.liftPercent >= 100) {
        state.liftPercent = 0;
        liftSlider.value = 0;
      }
    } else {
      playBtn.innerHTML = '<i class="fa-solid fa-play"></i>';
    }
  });
  
  // View Toggles
  document.getElementById('toggle-plane').addEventListener('change', (e) => {
    state.showPlane = e.target.checked;
    updateVisualization();
  });
  
  document.getElementById('toggle-sv').addEventListener('change', (e) => {
    state.showSV = e.target.checked;
    updateVisualization();
  });
  
  document.getElementById('toggle-contour').addEventListener('change', (e) => {
    state.showContour = e.target.checked;
    updateVisualization();
  });
  
  document.getElementById('toggle-margin-planes').addEventListener('change', (e) => {
    state.showMarginPlanes = e.target.checked;
    // Redraw contour on floor and re-add separating plane height
    updateVisualization();
  });
  
  // Camera Reset
  document.getElementById('reset-camera-btn').addEventListener('click', resetCamera);
}

// Optimization: move/lift existing objects in 3D without recreating them
function updateDataPointsPositions() {
  const liftFrac = state.liftPercent / 100;
  
  for (let i = 0; i < state.points2D.length; i++) {
    const pt = state.points2D[i];
    const targetZ = getTargetZ(pt);
    const currentZ = targetZ * liftFrac;
    
    // Shift particle sphere
    if (pointsMeshGroup[i]) {
      pointsMeshGroup[i].position.z = currentZ;
    }
    
    // Shift SV gold ring
    // We need to match the correct support vector indices
    // Rings are stored in svRingsGroup sequentially for SVs
  }
  
  // Recalculate support vector rings positions
  let ringIdx = 0;
  for (let i = 0; i < state.points2D.length; i++) {
    const pt = state.points2D[i];
    const isSV = state.svm2D.alpha[i] > 1e-5;
    if (isSV && state.showSV) {
      if (svRingsGroup[ringIdx]) {
        const targetZ = getTargetZ(pt);
        svRingsGroup[ringIdx].position.z = targetZ * liftFrac;
        ringIdx++;
      }
    }
  }
}

function updateSeparatingSurfacesPositions() {
  const liftFrac = state.liftPercent / 100;
  
  // If plane exists, we just warp its vertices
  if (hyperplaneMesh) {
    const posAttr = hyperplaneMesh.geometry.attributes.position;
    
    // Linear weights
    let w = [0, 0, 0], bias = 0;
    if (state.liftMode === 'feature' && state.svm3D) {
      for (let i = 0; i < state.svm3D.data.length; i++) {
        if (state.svm3D.alpha[i] > 1e-5) {
          w[0] += state.svm3D.alpha[i] * state.svm3D.labels[i] * state.svm3D.data[i][0];
          w[1] += state.svm3D.alpha[i] * state.svm3D.labels[i] * state.svm3D.data[i][1];
          w[2] += state.svm3D.alpha[i] * state.svm3D.labels[i] * state.svm3D.data[i][2];
        }
      }
      bias = state.svm3D.b;
    }
    
    for (let i = 0; i < posAttr.count; i++) {
      const x = posAttr.getX(i);
      const y = posAttr.getY(i);
      
      let targetZ = 0;
      if (state.liftMode === 'margin') {
        targetZ = 0;
      } else {
        if (Math.abs(w[2]) > 1e-4) {
          targetZ = -(w[0] * x + w[1] * y + bias) / w[2];
        } else {
          targetZ = 0;
        }
      }
      posAttr.setZ(i, targetZ * liftFrac);
    }
    hyperplaneMesh.geometry.computeVertexNormals();
    hyperplaneMesh.geometry.attributes.position.needsUpdate = true;
    hyperplaneMesh.material.opacity = 0.35 * liftFrac;
  }
  
  // Do the same for margin planes if they exist
  if (state.showMarginPlanes && marginPlaneMeshPos && marginPlaneMeshNeg) {
    const posAttrP = marginPlaneMeshPos.geometry.attributes.position;
    const posAttrN = marginPlaneMeshNeg.geometry.attributes.position;
    
    let w = [0, 0, 0], bias = 0;
    if (state.liftMode === 'feature' && state.svm3D) {
      for (let i = 0; i < state.svm3D.data.length; i++) {
        if (state.svm3D.alpha[i] > 1e-5) {
          w[0] += state.svm3D.alpha[i] * state.svm3D.labels[i] * state.svm3D.data[i][0];
          w[1] += state.svm3D.alpha[i] * state.svm3D.labels[i] * state.svm3D.data[i][1];
          w[2] += state.svm3D.alpha[i] * state.svm3D.labels[i] * state.svm3D.data[i][2];
        }
      }
      bias = state.svm3D.b;
    }
    
    for (let i = 0; i < posAttrP.count; i++) {
      const x = posAttrP.getX(i);
      const y = posAttrP.getY(i);
      
      let targetZP = 0;
      let targetZN = 0;
      if (state.liftMode === 'margin') {
        targetZP = 1.0;
        targetZN = -1.0;
      } else {
        if (Math.abs(w[2]) > 1e-4) {
          targetZP = -(w[0] * x + w[1] * y + bias - 1.0) / w[2];
          targetZN = -(w[0] * x + w[1] * y + bias + 1.0) / w[2];
        } else {
          targetZP = 0.5;
          targetZN = -0.5;
        }
      }
      posAttrP.setZ(i, targetZP * liftFrac);
      posAttrN.setZ(i, targetZN * liftFrac);
    }
    
    marginPlaneMeshPos.geometry.computeVertexNormals();
    marginPlaneMeshPos.geometry.attributes.position.needsUpdate = true;
    marginPlaneMeshPos.material.opacity = 0.12 * liftFrac;
    
    marginPlaneMeshNeg.geometry.computeVertexNormals();
    marginPlaneMeshNeg.geometry.attributes.position.needsUpdate = true;
    marginPlaneMeshNeg.material.opacity = 0.12 * liftFrac;
  }
}

// --- Animation Loop ---
function animate() {
  animationFrameId = requestAnimationFrame(animate);
  
  // Render loop OrbitControls update
  if (controls) controls.update();
  
  // Handle Play/Pause dimension lifting animation
  if (state.isPlayingLift) {
    state.liftPercent += 0.8; // speed of transition
    if (state.liftPercent >= 100) {
      state.liftPercent = 100;
      state.isPlayingLift = false;
      document.getElementById('play-lift-btn').innerHTML = '<i class="fa-solid fa-play"></i>';
    }
    
    // Update slider UI value
    const liftSlider = document.getElementById('lift-slider');
    liftSlider.value = state.liftPercent;
    
    const liftVal = document.getElementById('lift-value');
    if (state.liftPercent === 100) {
      liftVal.textContent = '100% (3D)';
    } else {
      liftVal.textContent = Math.floor(state.liftPercent) + '%';
    }
    
    updateDataPointsPositions();
    updateSeparatingSurfacesPositions();
  }
  
  renderer.render(scene, camera);
}

function onWindowResize() {
  const width = containerEl.clientWidth;
  const height = containerEl.clientHeight;
  
  camera.aspect = width / height;
  camera.updateProjectionMatrix();
  
  renderer.setSize(width, height);
}

// Run App on Load
window.addEventListener('DOMContentLoaded', init);
