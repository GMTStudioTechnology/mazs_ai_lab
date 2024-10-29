export async function processImageFile(imageFile: ArrayBuffer): Promise<string> {
    return new Promise(async (resolve, reject) => {
      const canvas = document.createElement('canvas');
      const ctx = canvas.getContext('2d');
      if (!ctx) {
        reject('Failed to get canvas context');
        return;
      }
  
      const img = new Image();
      const url = URL.createObjectURL(new Blob([imageFile]));
      img.src = url;
  
      img.onload = async () => {
        try {
          canvas.width = img.width;
          canvas.height = img.height;
          ctx.drawImage(img, 0, 0, img.width, img.height);
          const imageData = ctx.getImageData(0, 0, img.width, img.height);
          const data = imageData.data;
  
          // Calculate average color and color histogram
          const colorHistogram = new Array(256).fill(0);
          let r = 0, g = 0, b = 0;
          for (let i = 0; i < data.length; i += 4) {
            r += data[i];
            g += data[i + 1];
            b += data[i + 2];
            const brightness = Math.round((data[i] + data[i + 1] + data[i + 2]) / 3);
            colorHistogram[brightness]++;
          }
          const pixelCount = data.length / 4;
          r = Math.floor(r / pixelCount);
          g = Math.floor(g / pixelCount);
          b = Math.floor(b / pixelCount);
  
          // Detect edges using improved Sobel operator
          const sobelData = applyImprovedSobelOperator(imageData);
          const edgePercentage = calculateEdgePercentage(sobelData);
  
          // Analyze color distribution with more granularity
          const colorDistribution = analyzeDetailedColorDistribution(data);
  
          // Estimate image complexity using multiple factors
          const complexity = estimateAdvancedComplexity(sobelData, colorDistribution, colorHistogram);
  
          // Detect shapes with improved categorization
          const shapes = detectAdvancedShapes(sobelData, imageData.width, imageData.height);
  
          // Analyze composition using rule of thirds and golden ratio
          const composition = analyzeAdvancedComposition(imageData);
  
          // Analyze texture
          const texture = analyzeTexture(sobelData, imageData.width, imageData.height);
  
          // Detect humans and count them
          const humanDetection = detectHumansVid(imageData);
  
          // Generate enhanced image description
          const description = generateComprehensiveImageDescription(
            r, g, b, edgePercentage, colorDistribution, complexity, shapes, composition, texture, humanDetection
          );
  
          // Perform OCR to recognize text in the image
          const detectedText = await performOCR(canvas);
  
          // Combine image description with detected text
          const finalDescription = detectedText
            ? `${description}\n\nDetected Text: "${detectedText}"`
            : description;
  
          resolve(finalDescription);
          URL.revokeObjectURL(url);
        } catch (error) {
          reject(error);
          URL.revokeObjectURL(url);
        }
      };
  
      img.onerror = () => {
        reject('Failed to load image');
        URL.revokeObjectURL(url);
      };
    });
  }
  
  // Function to perform OCR using Tesseract.js without npm install
  async function performOCR(canvas: HTMLCanvasElement): Promise<string> {
    // Dynamically load Tesseract.js from CDN if not already loaded
    if (!(window as any).Tesseract) {
      try {
        await loadScript('https://cdn.jsdelivr.net/npm/tesseract.js@2/dist/tesseract.min.js');
        if (!(window as any).Tesseract) {
          throw new Error('Failed to load Tesseract.js');
        }
      } catch (error) {
        console.error('Error loading Tesseract.js:', error);
        throw error;
      }
    }
  
    const { Tesseract } = (window as any);
    const worker = Tesseract.createWorker({
      logger: (m: any) => console.log(m), // Optional: Remove or modify for less verbosity
      workerPath: 'https://cdn.jsdelivr.net/npm/tesseract.js@2/dist/worker.min.js',
      corePath: 'https://cdn.jsdelivr.net/npm/tesseract.js-core@2/tesseract-core.wasm.js',
    });
  
    try {
      await worker.load();
      await worker.loadLanguage('eng');
      await worker.initialize('eng');
  
      // Preprocess the image for better OCR results
      const preprocessedCanvas = await preprocessImage(canvas);
  
      const { data: { text, confidence } } = await worker.recognize(preprocessedCanvas);
  
      console.log(`OCR Confidence: ${confidence}`);
  
      // Post-process the recognized text
      const processedText = postprocessText(text);
  
      return processedText;
    } catch (error) {
      console.error('Error during OCR process:', error);
      throw error;
    } finally {
      await worker.terminate();
    }
  }
  
  async function preprocessImage(canvas: HTMLCanvasElement): Promise<HTMLCanvasElement> {
    const ctx = canvas.getContext('2d');
    if (!ctx) throw new Error('Unable to get 2D context');
  
    // Apply image preprocessing techniques
    // 1. Convert to grayscale
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const data = imageData.data;
    for (let i = 0; i < data.length; i += 4) {
      const avg = (data[i] + data[i + 1] + data[i + 2]) / 3;
      data[i] = data[i + 1] = data[i + 2] = avg;
    }
    ctx.putImageData(imageData, 0, 0);
  
    // 2. Increase contrast
    ctx.filter = 'contrast(150%)';
    ctx.drawImage(canvas, 0, 0);
    ctx.filter = 'none';
  
    // 3. Apply thresholding
    const thresholdImageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const thresholdData = thresholdImageData.data;
    const threshold = 128;
    for (let i = 0; i < thresholdData.length; i += 4) {
      const value = thresholdData[i] > threshold ? 255 : 0;
      thresholdData[i] = thresholdData[i + 1] = thresholdData[i + 2] = value;
    }
    ctx.putImageData(thresholdImageData, 0, 0);
  
    return canvas;
  }
  
  function postprocessText(text: string): string {
    // Remove extra whitespace
    let processedText = text.replace(/\s+/g, ' ').trim();
  
    // Correct common OCR mistakes
    const corrections: { [key: string]: string } = {
      '0': 'O',
      '1': 'I',
      '5': 'S',
      '8': 'B',
      // Add more corrections as needed
    };
  
    for (const [mistake, correction] of Object.entries(corrections)) {
      processedText = processedText.replace(new RegExp(mistake, 'g'), correction);
    }
  
    return processedText;
  }
  
  // Utility function to dynamically load a script
  function loadScript(src: string): Promise<void> {
    return new Promise((resolve, reject) => {
      // Check if the script is already loaded
      if (document.querySelector(`script[src="${src}"]`)) {
        resolve();
        return;
      }
  
      const script = document.createElement('script');
      script.src = src;
      script.async = true;
  
      script.onload = () => {
        resolve();
      };
  
      script.onerror = () => {
        reject(new Error(`Failed to load script ${src}`));
      };
  
      document.head.appendChild(script);
    });
  }
  
  function applyImprovedSobelOperator(imageData: ImageData): Uint8ClampedArray {
    const width = imageData.width;
    const height = imageData.height;
    const data = imageData.data;
    const sobelData = new Uint8ClampedArray(width * height * 4);
  
    const sobelX = [-1, 0, 1, -2, 0, 2, -1, 0, 1];
    const sobelY = [-1, -2, -1, 0, 0, 0, 1, 2, 1];
  
    for (let y = 1; y < height - 1; y++) {
      for (let x = 1; x < width - 1; x++) {
        let gx = 0, gy = 0;
        for (let j = -1; j <= 1; j++) {
          for (let i = -1; i <= 1; i++) {
            const idx = ((y + j) * width + (x + i)) * 4;
            const gray = (data[idx] + data[idx + 1] + data[idx + 2]) / 3;
            gx += gray * sobelX[(j + 1) * 3 + (i + 1)];
            gy += gray * sobelY[(j + 1) * 3 + (i + 1)];
          }
        }
        const magnitude = Math.sqrt(gx * gx + gy * gy);
        const idx = (y * width + x) * 4;
        sobelData[idx] = sobelData[idx + 1] = sobelData[idx + 2] = magnitude;
        sobelData[idx + 3] = 255;
      }
    }
  
    return sobelData;
  }
  
  function calculateEdgePercentage(sobelData: Uint8ClampedArray): number {
    const threshold = 50;
    let edgePixels = 0;
    for (let i = 0; i < sobelData.length; i += 4) {
      if (sobelData[i] > threshold) {
        edgePixels++;
      }
    }
    return (edgePixels / (sobelData.length / 4)) * 100;
  }
  
  function analyzeDetailedColorDistribution(data: Uint8ClampedArray): { [key: string]: number } {
    const colorBuckets: { [key: string]: number } = {
      red: 0, darkRed: 0, lightRed: 0,
      green: 0, darkGreen: 0, lightGreen: 0,
      blue: 0, darkBlue: 0, lightBlue: 0,
      yellow: 0, cyan: 0, magenta: 0,
      orange: 0, purple: 0, pink: 0,
      white: 0, black: 0, gray: 0,
      brown: 0, tan: 0, maroon: 0,
      navy: 0, olive: 0, teal: 0
    };
  
    for (let i = 0; i < data.length; i += 4) {
      const r = data[i];
      const g = data[i + 1];
      const b = data[i + 2];
  
      if (r > 240 && g > 240 && b > 240) colorBuckets.white++;
      else if (r < 15 && g < 15 && b < 15) colorBuckets.black++;
      else if (Math.abs(r - g) < 10 && Math.abs(g - b) < 10 && Math.abs(r - b) < 10) {
        colorBuckets.gray++;
      }
      else {
        const max = Math.max(r, g, b);
        const min = Math.min(r, g, b);
        const chroma = max - min;
        const hue = chroma === 0 ? 0 :
          max === r ? ((g - b) / chroma + 6) % 6 :
          max === g ? ((b - r) / chroma + 2) :
          ((r - g) / chroma + 4);
        const lightness = (max + min) / 2;
        const saturation = chroma === 0 ? 0 : chroma / (1 - Math.abs(2 * lightness - 1));
  
        if (saturation < 0.1) {
          colorBuckets.gray++;
        } else {
          if (hue < 0.5 || hue >= 5.5) {
            lightness < 0.5 ? colorBuckets.darkRed++ : colorBuckets.lightRed++;
          } else if (hue < 1.5) {
            colorBuckets.orange++;
          } else if (hue < 2.5) {
            lightness < 0.5 ? colorBuckets.darkGreen++ : colorBuckets.lightGreen++;
          } else if (hue < 3.5) {
            colorBuckets.cyan++;
          } else if (hue < 4.5) {
            lightness < 0.5 ? colorBuckets.darkBlue++ : colorBuckets.lightBlue++;
          } else {
            colorBuckets.purple++;
          }
        }
      }
    }
  
    const totalPixels = data.length / 4;
    Object.keys(colorBuckets).forEach(key => {
      colorBuckets[key] = (colorBuckets[key] / totalPixels) * 100;
    });
  
    return colorBuckets;
  }
  
  function estimateAdvancedComplexity(sobelData: Uint8ClampedArray, colorDistribution: { [key: string]: number }, colorHistogram: number[]): number {
    const edgeComplexity = calculateEdgePercentage(sobelData) / 100;
    const colorComplexity = 1 - Math.max(...Object.values(colorDistribution)) / 100;
    
    // Calculate color variety
    const nonZeroColors = colorHistogram.filter(count => count > 0).length;
    const colorVariety = nonZeroColors / 256;
  
    // Calculate histogram entropy
    const histogramEntropy = calculateHistogramEntropy(colorHistogram);
  
    return (edgeComplexity * 0.4 + colorComplexity * 0.3 + colorVariety * 0.15 + histogramEntropy * 0.15);
  }
  
  function calculateHistogramEntropy(histogram: number[]): number {
    const totalPixels = histogram.reduce((sum, count) => sum + count, 0);
    let entropy = 0;
    histogram.forEach(count => {
      if (count > 0) {
        const p = count / totalPixels;
        entropy -= p * Math.log2(p);
      }
    });
    return entropy / Math.log2(256); // Normalize to [0, 1]
  }
  
  function detectAdvancedShapes(sobelData: Uint8ClampedArray, width: number, height: number): { [key: string]: number } {
    const shapes: { [key: string]: number } = {
      horizontalLines: 0,
      verticalLines: 0,
      diagonalLines: 0,
      circles: 0
    };
  
    const houghSpace: number[][] = new Array(180).fill(0).map(() => new Array(Math.ceil(Math.sqrt(width * width + height * height))).fill(0));
  
    // Hough transform for line detection
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        if (sobelData[(y * width + x) * 4] > 50) {
          for (let theta = 0; theta < 180; theta++) {
            const r = Math.round(x * Math.cos(theta * Math.PI / 180) + y * Math.sin(theta * Math.PI / 180));
            if (r >= 0 && r < houghSpace[0].length) {
              houghSpace[theta][r]++;
            }
          }
        }
      }
    }
  
    // Detect lines
    const lineThreshold = Math.max(width, height) / 4;
    for (let theta = 0; theta < 180; theta++) {
      for (let r = 0; r < houghSpace[0].length; r++) {
        if (houghSpace[theta][r] > lineThreshold) {
          if (theta < 10 || theta > 170) shapes.verticalLines++;
          else if (theta > 80 && theta < 100) shapes.horizontalLines++;
          else shapes.diagonalLines++;
        }
      }
    }
  
    // Simple circle detection
    const centerX = Math.floor(width / 2);
    const centerY = Math.floor(height / 2);
    let circleVotes = 0;
    const radius = Math.min(width, height) / 4;
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        if (sobelData[(y * width + x) * 4] > 50) {
          const distance = Math.sqrt((x - centerX) ** 2 + (y - centerY) ** 2);
          if (Math.abs(distance - radius) < 5) circleVotes++;
        }
      }
    }
    if (circleVotes > Math.PI * radius) shapes.circles++;
  
    return shapes;
  }
  
  function analyzeAdvancedComposition(imageData: ImageData): string {
    const width = imageData.width;
    const height = imageData.height;
    const data = imageData.data;
  
    const thirdW = Math.floor(width / 3);
    const thirdH = Math.floor(height / 3);
    const goldenRatioW = Math.floor(width * 0.618);
    const goldenRatioH = Math.floor(height * 0.618);
  
    const regions = [
      {name: "top-left", x: 0, y: 0, w: thirdW, h: thirdH},
      {name: "top-center", x: thirdW, y: 0, w: thirdW, h: thirdH},
      {name: "top-right", x: 2 * thirdW, y: 0, w: width - 2 * thirdW, h: thirdH},
      {name: "middle-left", x: 0, y: thirdH, w: thirdW, h: thirdH},
      {name: "center", x: thirdW, y: thirdH, w: thirdW, h: thirdH},
      {name: "middle-right", x: 2 * thirdW, y: thirdH, w: width - 2 * thirdW, h: thirdH},
      {name: "bottom-left", x: 0, y: 2 * thirdH, w: thirdW, h: height - 2 * thirdH},
      {name: "bottom-center", x: thirdW, y: 2 * thirdH, w: thirdW, h: height - 2 * thirdH},
      {name: "bottom-right", x: 2 * thirdW, y: 2 * thirdH, w: width - 2 * thirdW, h: height - 2 * thirdH},
      {name: "golden-left", x: width - goldenRatioW, y: 0, w: 1, h: height},
      {name: "golden-right", x: goldenRatioW, y: 0, w: 1, h: height},
      {name: "golden-top", x: 0, y: height - goldenRatioH, w: width, h: 1},
      {name: "golden-bottom", x: 0, y: goldenRatioH, w: width, h: 1}
    ];
  
    const regionIntensities = regions.map(region => {
      let sum = 0;
      for (let y = region.y; y < region.y + region.h; y++) {
        for (let x = region.x; x < region.x + region.w; x++) {
          const idx = (y * width + x) * 4;
          sum += (data[idx] + data[idx + 1] + data[idx + 2]) / 3;
        }
      }
      return {name: region.name, intensity: sum / (region.w * region.h)};
    });
  
    regionIntensities.sort((a, b) => b.intensity - a.intensity);
    const topRegions = regionIntensities.slice(0, 3);
  
    let compositionDescription = "The image composition follows ";
    if (topRegions.some(r => r.name.includes("golden"))) {
      compositionDescription += "the golden ratio, ";
    }
    if (topRegions.some(r => !r.name.includes("golden"))) {
      compositionDescription += "the rule of thirds, ";
    }
    compositionDescription += `with the main focus areas in the ${topRegions.map(r => r.name).join(", ")} regions.`;
  
    return compositionDescription;
  }
  
  function analyzeTexture(sobelData: Uint8ClampedArray, width: number, height: number): string {
    let smoothCount = 0;
    let roughCount = 0;
    let edgeCount = 0;
  
    for (let i = 0; i < sobelData.length; i += 4) {
      const edgeStrength = sobelData[i];
      if (edgeStrength < 20) smoothCount++;
      else if (edgeStrength > 100) roughCount++;
      else edgeCount++;
    }
  
    const totalPixels = width * height;
    const smoothPercentage = (smoothCount / totalPixels) * 100;
    const roughPercentage = (roughCount / totalPixels) * 100;
    const edgePercentage = (edgeCount / totalPixels) * 100;
  
    let textureDescription = "";
    if (smoothPercentage > 60) textureDescription = "predominantly smooth";
    else if (roughPercentage > 40) textureDescription = "predominantly rough or detailed";
    else if (edgePercentage > 40) textureDescription = "balanced mix of smooth and detailed areas";
    else textureDescription = "varied texture with a mix of smooth and rough areas";
  
    return `The image has a ${textureDescription} texture.`;
  }
  
  function detectHumansVid(imageData: ImageData): { isHuman: boolean; count: number; confidence: number } {
    const data = imageData.data;
    const width = imageData.width;
    const height = imageData.height;
    let skinPixelCount = 0;
    let faceCount = 0;
    const totalPixels = data.length / 4;
  
    // Improved skin color detection using multiple color spaces
    function isSkinColor(r: number, g: number, b: number): boolean {
      const ycbcr = rgbToYCbCr(r, g, b);
      const hsv = rgbToHSV(r, g, b);
      return (
        (ycbcr.cb > 77 && ycbcr.cb < 127 && ycbcr.cr > 133 && ycbcr.cr < 173) &&
        (hsv.h >= 0 && hsv.h <= 50 && hsv.s >= 0.23 && hsv.s <= 0.68)
      );
    }
  
    function rgbToYCbCr(r: number, g: number, b: number): { y: number; cb: number; cr: number } {
      const y = 0.299 * r + 0.587 * g + 0.114 * b;
      const cb = 128 - 0.168736 * r - 0.331264 * g + 0.5 * b;
      const cr = 128 + 0.5 * r - 0.418688 * g - 0.081312 * b;
      return { y, cb, cr };
    }
  
    function rgbToHSV(r: number, g: number, b: number): { h: number; s: number; v: number } {
      r /= 255; g /= 255; b /= 255;
      const max = Math.max(r, g, b);
      const min = Math.min(r, g, b);
      const d = max - min;
      let h = 0;
      const s = max === 0 ? 0 : d / max;
      const v = max;
  
      if (max !== min) {
        switch (max) {
          case r: h = (g - b) / d + (g < b ? 6 : 0); break;
          case g: h = (b - r) / d + 2; break;
          case b: h = (r - g) / d + 4; break;
        }
        h /= 6;
      }
  
      return { h: h * 360, s, v };
    }
  
    // Enhanced face detection using Haar-like features and additional checks
    function detectFace(startX: number, startY: number, windowSize: number): boolean {
      const features = [
        { type: 'eyes', y: 0.2, x: 0.2, w: 0.6, h: 0.2 },
        { type: 'nose', y: 0.4, x: 0.3, w: 0.4, h: 0.2 },
        { type: 'mouth', y: 0.6, x: 0.25, w: 0.5, h: 0.2 },
        { type: 'forehead', y: 0.05, x: 0.2, w: 0.6, h: 0.15 },
        { type: 'chin', y: 0.8, x: 0.25, w: 0.5, h: 0.15 },
      ];
  
      let featureScores: { [key: string]: number } = {};
  
      for (const feature of features) {
        let lightArea = 0;
        let darkArea = 0;
  
        for (let y = startY + feature.y * windowSize; y < startY + (feature.y + feature.h) * windowSize; y++) {
          for (let x = startX + feature.x * windowSize; x < startX + (feature.x + feature.w) * windowSize; x++) {
            const idx = (y * width + x) * 4;
            const brightness = (data[idx] + data[idx + 1] + data[idx + 2]) / 3;
  
            if (feature.type === 'eyes' || feature.type === 'mouth') {
              if (x < startX + (feature.x + feature.w / 2) * windowSize) {
                darkArea += brightness;
              } else {
                lightArea += brightness;
              }
            } else {
              if (y < startY + (feature.y + feature.h / 2) * windowSize) {
                lightArea += brightness;
              } else {
                darkArea += brightness;
              }
            }
          }
        }
  
        featureScores[feature.type] = Math.abs(lightArea - darkArea) / (lightArea + darkArea);
      }
  
      // Additional checks for face symmetry and proportions
      const symmetryScore = Math.abs(featureScores.eyes - featureScores.mouth);
      const proportionScore = Math.abs(featureScores.forehead - featureScores.chin);
  
      return (
        featureScores.eyes > 0.1 &&
        featureScores.nose > 0.05 &&
        featureScores.mouth > 0.08 &&
        featureScores.forehead > 0.03 &&
        featureScores.chin > 0.03 &&
        symmetryScore < 0.1 &&
        proportionScore < 0.15
      );
    }
  
    // Multi-scale face detection with improved step size
    const scales = [0.5, 0.75, 1, 1.25, 1.5];
    const faceDetections: { x: number; y: number; size: number }[] = [];
  
    for (const scale of scales) {
      const stepSize = Math.max(1, Math.floor(10 * scale));
      const windowSize = Math.floor(60 * scale);
  
      for (let y = 0; y < height - windowSize; y += stepSize) {
        for (let x = 0; x < width - windowSize; x += stepSize) {
          if (detectFace(x, y, windowSize)) {
            faceDetections.push({ x, y, size: windowSize });
          }
        }
      }
    }
  
    // Non-maximum suppression to remove overlapping detections
    const finalFaces = nonMaxSuppression(faceDetections, 0.3);
    faceCount = Math.floor(finalFaces.length * 0.2 / 100);
  
    // Calculate skin pixel percentage with improved thresholds
    for (let i = 0; i < data.length; i += 4) {
      if (isSkinColor(data[i], data[i + 1], data[i + 2])) {
        skinPixelCount++;
      }
    }
  
    const skinPercentage = (skinPixelCount / totalPixels) * 100;
    const isHuman = faceCount > 0 || skinPercentage > 15;
    const confidence = Math.min((skinPercentage / 3) + (faceCount * 20), 100);
  
    return { isHuman, count: faceCount, confidence };
  }
  
  function nonMaxSuppression(boxes: { x: number; y: number; size: number }[], overlapThresh: number): { x: number; y: number; size: number }[] {
    if (boxes.length === 0) return [];
  
    const pick: number[] = [];
    const x1 = boxes.map(box => box.x);
    const y1 = boxes.map(box => box.y);
    const x2 = boxes.map(box => box.x + box.size);
    const y2 = boxes.map(box => box.y + box.size);
    const area = boxes.map(box => box.size * box.size);
    const idxs = Array.from(Array(boxes.length).keys()).sort((a, b) => area[b] - area[a]);
  
    while (idxs.length > 0) {
      const last = idxs.length - 1;
      const i = idxs[last];
      pick.push(i);
  
      const xx1 = idxs.map(j => Math.max(x1[i], x1[j]));
      const yy1 = idxs.map(j => Math.max(y1[i], y1[j]));
      const xx2 = idxs.map(j => Math.min(x2[i], x2[j]));
      const yy2 = idxs.map(j => Math.min(y2[i], y2[j]));
  
      const w = xx2.map((x, j) => Math.max(0, x - xx1[j]));
      const h = yy2.map((y, j) => Math.max(0, y - yy1[j]));
  
      const overlap = w.map((ww, j) => (ww * h[j]) / area[idxs[j]]);
  
      idxs.splice(last, 1);
      for (let j = overlap.length - 1; j >= 0; j--) {
        if (overlap[j] > overlapThresh) {
          idxs.splice(j, 1);
        }
      }
    }
  
    return pick.map(i => boxes[i]);
  }
  
  function generateComprehensiveImageDescription(r: number, g: number, b: number, edgePercentage: number, colorDistribution: { [key: string]: number }, complexity: number, shapes: { [key: string]: number }, composition: string, texture: string, humanDetection: { isHuman: boolean; count: number }): string {
    const dominantColors = Object.entries(colorDistribution)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 3)
      .map(([color, percentage]) => `${color} (${percentage.toFixed(1)}%)`);
  
    const complexityDescription = complexity < 0.3 ? "simple" : complexity < 0.7 ? "moderately complex" : "very complex";
  
    const shapeDescription = Object.entries(shapes)
      .filter(([_, count]) => count > 0)
      .map(([shape, count]) => `${count} ${shape}`)
      .join(", ");
  
    const humanDescription = humanDetection.isHuman
      ? `The image likely contains ${humanDetection.count} human${humanDetection.count > 1 ? 's' : ''}.`
      : "No humans were detected in the image.";
  
    return `This image is a ${complexityDescription} composition with an average color of rgb(${r}, ${g}, ${b}).
    The dominant colors are ${dominantColors.join(", ")}, creating a ${complexity > 0.5 ? "vibrant and diverse" : "harmonious and focused"} visual palette.
    ${composition}
    ${texture}
    The image contains approximately ${edgePercentage.toFixed(2)}% edge pixels, indicating ${edgePercentage > 20 ? "a high level of detail" : "a relatively smooth overall appearance"}.
    ${shapeDescription ? `Detected shapes include ${shapeDescription}.` : "No distinct shapes were detected."}
    ${humanDescription}
    Overall, this image presents a ${complexityDescription} visual experience with ${dominantColors[0].split(" ")[0]} as the primary color, featuring ${texture.toLowerCase()} and a ${composition.toLowerCase()}`;
  }
  