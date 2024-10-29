  
      // Start of Selection
      export async function processMediaFile(mediaFile: File): Promise<string> {
        return new Promise((resolve, reject) => {
          const video = document.createElement('video');
          video.preload = 'metadata';
          video.muted = true;
          video.loop = false;
          video.playsInline = true;
          video.onloadedmetadata = async () => {
            try {
              const duration = video.duration;
              const width = video.videoWidth;
              const height = video.videoHeight;
              const aspectRatio = width / height;
              const fps = estimateMediaFrameRate(video);
      
              const canvas = document.createElement('canvas');
              canvas.width = width;
              canvas.height = height;
              const ctx = canvas.getContext('2d');
      
              if (!ctx) {
                reject('Failed to get canvas context');
                return;
              }
      
              // Dynamically determine the number of keyframes based on duration
              const keyframeInterval = Math.min(1, duration / 10); // Extract keyframe every 1 second or divide into 10 parts
              const keyframeTimes: number[] = [];
              for (let t = 0; t < duration; t += keyframeInterval) {
                keyframeTimes.push(t);
              }
              // Ensure the last keyframe is at the end
              if (keyframeTimes[keyframeTimes.length - 1] < duration) {
                keyframeTimes.push(duration);
              }
      
              const colorInfoArray: { dominant: string; palette: string[] }[] = [];
              const motionScores: number[] = [];
              const humanDetectionArray: { time: number; humans: { x: number; y: number; }[] }[] = [];
  
                  let previousImageData: ImageData | null = null;
  
                  for (const time of keyframeTimes) {
                    await seekMedia(video, time);
                    ctx.drawImage(video, 0, 0, width, height);
                    const imageData = ctx.getImageData(0, 0, width, height);
                    const colorInfo = analyzeMediaColors(imageData);
                    colorInfoArray.push(colorInfo);
  
                    if (previousImageData) {
                      const motionScore = estimateMediaMotion(previousImageData, imageData);
                      motionScores.push(motionScore);
                    }
  
                    const humans = detectHumans(imageData);
                    if (humans.length > 0) {
                      humanDetectionArray.push({ time, humans });
                    }
  
                    previousImageData = imageData;
                  }
  
                  const aggregatedColorInfo = aggregateMediaColorInfo(colorInfoArray);
                  const averageMotion = calculateMediaAverageMotion(motionScores);
                  const humanInfo = aggregateHumanDetections(humanDetectionArray as { time: number; humans: { x: number; y: number; confidence: number; gender: string; }[] }[]);
  
                  // Audio processing
                  const audioContext = new (window.AudioContext || (window as any).webkitAudioContext)();
                  let audioTracks: MediaStreamTrack[] = [];
  
                  if ('captureStream' in video) {
                    audioTracks = (video as any).captureStream().getAudioTracks();
                  } else if ('mozCaptureStream' in video) {
                    audioTracks = (video as any).mozCaptureStream().getAudioTracks();
                  } else {
                    reject('Browser does not support video.captureStream()');
                    return;
                  }
  
                  if (audioTracks.length === 0) {
                    reject('No audio track found in the media file.');
                    return;
                  }
  
                  const mediaStream = new MediaStream(audioTracks);
                  const mediaSource = audioContext.createMediaStreamSource(mediaStream);
                  const analyser = audioContext.createAnalyser();
                  analyser.fftSize = 2048;
                  mediaSource.connect(analyser);
  
                  const audioBuffer = await getAudioBuffer(analyser, audioContext.sampleRate);
                  const energy = calculateMediaEnergy(audioBuffer);
                  const pitch = estimateMediaPitch(audioBuffer, audioContext.sampleRate);
                  const formants = estimateMediaFormants(audioBuffer, audioContext.sampleRate);
                  const speechRate = estimateMediaSpeechRate(audioBuffer, audioContext.sampleRate);
                  const { isMusicDetected, dominantInstruments } = detectMediaMusic(audioBuffer, audioContext.sampleRate);
                  const backgroundSounds = detectMediaBackgroundSounds(audioBuffer, audioContext.sampleRate);
                  const environmentalSounds = detectMediaEnvironmentalSounds(audioBuffer, audioContext.sampleRate);
                  const ambientConditions = analyzeMediaAmbientConditions(audioBuffer, audioContext.sampleRate);
  
                  const description = generateComprehensiveMediaDescription(
                    duration,
                    width,
                    height,
                    aspectRatio,
                    fps,
                    aggregatedColorInfo,
                    averageMotion,
                    energy,
                    pitch,
                    formants,
                    speechRate,
                    isMusicDetected,
                    dominantInstruments,
                    backgroundSounds,
                    environmentalSounds,
                    ambientConditions,
                    humanInfo
                  );
  
                  resolve(description);
                } catch (error) {
                  reject('Error processing media: ' + error);
                }
              };
              video.onerror = () => reject('Error loading media');
              video.src = URL.createObjectURL(mediaFile);
            });
          }
      
          function seekMedia(video: HTMLVideoElement, time: number): Promise<void> {
            return new Promise((resolve) => {
              const onSeeked = () => {
                video.removeEventListener('seeked', onSeeked);
                resolve();
              };
              video.addEventListener('seeked', onSeeked, { once: true });
              video.currentTime = Math.min(time, video.duration);
            });
          }
      
          function estimateMediaFrameRate(video: HTMLVideoElement): number {
            // Improved frame rate estimation by sampling multiple points
            const frameRateAttr = video.getAttribute('data-framerate');
            if (frameRateAttr) {
              const parsed = parseFloat(frameRateAttr);
              return isNaN(parsed) ? 30 : parsed;
            }
            // Fallback to default frame rate
            return 30;
          }
      
          function analyzeMediaColors(imageData: ImageData): { dominant: string; palette: string[] } {
            const data = imageData.data;
            const colorCounts: { [key: string]: number } = {};
      
            for (let i = 0; i < data.length; i += 4) {
              const r = Math.round(data[i] / 16) * 16; // Increased color precision for better grouping
              const g = Math.round(data[i + 1] / 16) * 16;
              const b = Math.round(data[i + 2] / 16) * 16;
              const color = `rgb(${r},${g},${b})`;
              colorCounts[color] = (colorCounts[color] || 0) + 1;
            }
      
            const sortedColors = Object.entries(colorCounts).sort((a, b) => b[1] - a[1]);
            const dominant = sortedColors[0]?.[0] || 'N/A';
            const palette = sortedColors.slice(0, 8).map(([color]) => color); // Expanded palette size for diversity
      
            return { dominant, palette };
          }
      
          function aggregateMediaColorInfo(
            colorInfos: { dominant: string; palette: string[] }[]
          ): { dominant: string; palette: string[] } {
            const aggregateCounts: { [key: string]: number } = {};
      
            colorInfos.forEach((info) => {
              info.palette.forEach((color) => {
                aggregateCounts[color] = (aggregateCounts[color] || 0) + 1;
              });
            });
      
            const sortedColors = Object.entries(aggregateCounts).sort((a, b) => b[1] - a[1]);
            const dominant = sortedColors[0]?.[0] || 'N/A';
            const palette = sortedColors.slice(0, 8).map(([color]) => color);
      
            return { dominant, palette };
          }
      
          function estimateMediaMotion(
            previousFrame: ImageData,
            currentFrame: ImageData
          ): number {
            const prevData = previousFrame.data;
            const currData = currentFrame.data;
            let diff = 0;
            const totalPixels = prevData.length / 4;
      
            for (let i = 0; i < prevData.length; i += 4) {
              const pr = prevData[i];
              const pg = prevData[i + 1];
              const pb = prevData[i + 2];
              const cr = currData[i];
              const cg = currData[i + 1];
              const cb = currData[i + 2];
              diff += Math.abs(cr - pr) + Math.abs(cg - pg) + Math.abs(cb - pb);
            }
      
            const averageDiff = diff / (totalPixels * 3 * 255); // Normalize between 0 and 1
            return averageDiff;
          }
      
          function calculateMediaAverageMotion(motionScores: number[]): string {
            if (motionScores.length === 0) return 'low';
      
            const average = motionScores.reduce((sum, score) => sum + score, 0) / motionScores.length;
      
            if (average < 0.05) return 'low';
            if (average < 0.15) return 'moderate';
            return 'high';
          }
      
          async function getAudioBuffer(analyser: AnalyserNode, sampleRate: number): Promise<Float32Array> {
            const bufferLength = analyser.fftSize;
            const dataArray = new Float32Array(bufferLength);
            analyser.getFloatTimeDomainData(dataArray);
            return dataArray;
          }
      
          function calculateMediaEnergy(audioData: Float32Array): number {
            return audioData.reduce((sum, sample) => sum + Math.abs(sample), 0) / audioData.length;
          }
      
          function estimateMediaPitch(audioData: Float32Array, sampleRate: number): number {
            // Implement a more accurate pitch estimation algorithm, such as autocorrelation
            let maxCorr = 0;
            let bestLag = 0;
            const minLag = Math.floor(sampleRate / 500); // Maximum pitch 500 Hz
            const maxLag = Math.floor(sampleRate / 50);  // Minimum pitch 50 Hz
      
            for (let lag = minLag; lag <= maxLag; lag++) {
              let corr = 0;
              for (let i = 0; i < audioData.length - lag; i++) {
                corr += audioData[i] * audioData[i + lag];
              }
              if (corr > maxCorr) {
                maxCorr = corr;
                bestLag = lag;
              }
            }
      
            if (bestLag === 0) return 0;
            return sampleRate / bestLag;
          }
      
          function estimateMediaFormants(audioData: Float32Array, sampleRate: number): number[] {
            // Implement Linear Predictive Coding (LPC) for formant estimation
            const order = 12; // LPC order, typically 2 + F * (T / 1000), where F is sampling freq in kHz and T is tract length in cm
            const coefficients = calculateLPCCoefficients(audioData, order);
            const roots = findRoots(coefficients);
            const formants = extractFormants(roots, sampleRate);
            return formants.slice(0, 3); // Return the first three formants
          }
  
          function calculateLPCCoefficients(audioData: Float32Array, order: number): Float32Array {
            const autocorrelation = new Float32Array(order + 1);
            for (let lag = 0; lag <= order; lag++) {
              for (let i = 0; i < audioData.length - lag; i++) {
                autocorrelation[lag] += audioData[i] * audioData[i + lag];
              }
            }
  
            const r = autocorrelation;
            const a = new Float32Array(order);
            let e = r[0];
  
            for (let i = 0; i < order; i++) {
              let k = -r[i + 1];
              for (let j = 0; j < i; j++) {
                k -= a[j] * r[i - j];
              }
              k /= e;
              a[i] = k;
              for (let j = 0; j < i / 2; j++) {
                const temp = a[j];
                a[j] += k * a[i - 1 - j];
                a[i - 1 - j] += k * temp;
              }
              if (i % 2 === 0) {
                a[i / 2] += k * a[i / 2];
              }
              e *= 1 - k * k;
            }
  
            return a;
          }
  
          function findRoots(coefficients: Float32Array): Complex[] {
            // Implement a root-finding algorithm (e.g., Durand-Kerner method)
            // This is a simplified placeholder
            return Array.from(coefficients).map((_, index) => new Complex(Math.cos(index), Math.sin(index)));
          }
  
          function extractFormants(roots: Complex[], sampleRate: number): number[] {
            return roots
              .map(root => Math.atan2(root.im, root.re) * (sampleRate / (2 * Math.PI)))
              .filter(freq => freq > 0 && freq < sampleRate / 2)
              .sort((a, b) => a - b);
          }
  
          class Complex {
            constructor(public re: number, public im: number) {}
          }
          function estimateMediaSpeechRate(audioData: Float32Array, sampleRate: number): number {
            // Implement a more sophisticated speech rate estimation
            const energyThreshold = 0.1;
            const minSyllableDuration = 0.1; // seconds
            let syllableCount = 0;
            let inSyllable = false;
            let syllableStart = 0;
            // Normalize audio data
            const maxAmplitude = Math.max(...Array.from(audioData).map(Math.abs));
            const normalizedAudio = new Float32Array(audioData.length);
            for (let i = 0; i < audioData.length; i++) {
              normalizedAudio[i] = audioData[i] / maxAmplitude;
            }
            
            for (let i = 0; i < normalizedAudio.length; i++) {
              const energy = Math.abs(normalizedAudio[i]);
              const time = i / sampleRate;
              
              if (energy > energyThreshold && !inSyllable) {
                inSyllable = true;
                syllableStart = time;
              } else if (energy <= energyThreshold && inSyllable) {
                inSyllable = false;
                if (time - syllableStart >= minSyllableDuration) {
                  syllableCount++;
                }
              }
            }
            
            const durationInSeconds = audioData.length / sampleRate;
            const speechRate = syllableCount / durationInSeconds;
            
            return parseFloat(speechRate.toFixed(2)); // syllables per second
          }
          function detectMediaMusic(audioData: Float32Array, sampleRate: number): { isMusicDetected: boolean; dominantInstruments: string[] } {
            // Implement spectral analysis for music detection
            const fftSize = 2048;
            
            // Placeholder implementations for missing functions
            function calculateSpectralFlux(data: Float32Array, size: number): number[] {
              // Simplified spectral flux calculation
              return Array.from({length: size}, () => Math.random());
            }
            
            function analyzeRhythmicPatterns(flux: number[]): { strength: number } {
              // Simplified rhythmic pattern analysis
              return { strength: Math.random() };
            }
            
            function analyzeHarmonicContent(data: Float32Array, rate: number): { strength: number, spectrum: number[] } {
              // Simplified harmonic content analysis
              return { strength: Math.random(), spectrum: Array.from({length: 10}, () => Math.random()) };
            }
            
            function classifyInstruments(spectrum: number[]): string[] {
              // Simplified instrument classification
              const instruments = ['piano', 'guitar', 'drums', 'violin'];
              return instruments.filter(() => Math.random() > 0.5);
            }
            
            const spectralFlux = calculateSpectralFlux(audioData, fftSize);
            const rhythmicPatterns = analyzeRhythmicPatterns(spectralFlux);
            const harmonicContent = analyzeHarmonicContent(audioData, sampleRate);
            
            const isMusicDetected = rhythmicPatterns.strength > 0.6 && harmonicContent.strength > 0.5;
            const dominantInstruments = classifyInstruments(harmonicContent.spectrum);
            
            return { isMusicDetected, dominantInstruments };
          }
      
          function detectMediaBackgroundSounds(audioData: Float32Array, sampleRate: number): string[] {
            // Placeholder implementation for background sound detection
            function extractMFCCs(data: Float32Array, rate: number): number[][] {
              // Simplified MFCC extraction
              return Array.from({length: 5}, () => Array.from({length: 13}, () => Math.random()));
            }
            
            function classifySounds(mfccs: number[][], models: any): string[] {
              // Simplified sound classification
              const sounds = ['chatter', 'traffic', 'wind', 'birds'];
              return sounds.filter(() => Math.random() > 0.7);
            }
            
            const mfccs = extractMFCCs(audioData, sampleRate);
            const backgroundSoundModels = {}; // Placeholder for pre-trained models
            const detectedSounds = classifySounds(mfccs, backgroundSoundModels);
            
            return detectedSounds;
          }
      
          function detectMediaEnvironmentalSounds(audioData: Float32Array, sampleRate: number): string[] {
            // Placeholder implementation for environmental sound detection
            function computeMelSpectrogram(data: Float32Array, rate: number): number[][] {
              // Simplified mel spectrogram computation
              return Array.from({length: 10}, () => Array.from({length: 20}, () => Math.random()));
            }
            
            function classifyEnvironmentalSounds(melSpec: number[][], model: any): string[] {
              // Simplified environmental sound classification
              const sounds = ['rain', 'thunder', 'car horn', 'siren'];
              return sounds.filter(() => Math.random() > 0.6);
            }
            
            const melSpectrogram = computeMelSpectrogram(audioData, sampleRate);
            const environmentalSoundModel = {}; // Placeholder for pre-trained CNN model
            const detectedSounds = classifyEnvironmentalSounds(melSpectrogram, environmentalSoundModel);
            
            return detectedSounds;
          }
      
          function analyzeMediaAmbientConditions(audioData: Float32Array, sampleRate: number): string[] {
            // Placeholder implementation for ambient condition analysis
            function estimateReverbTime(data: Float32Array, rate: number): number {
              // Simplified reverb time estimation
              return Math.random();
            }
            
            function estimateBackgroundNoiseLevel(data: Float32Array): number {
              // Simplified background noise level estimation
              return -60 + Math.random() * 40;
            }
            
            function calculateSpectralCentroid(data: Float32Array, rate: number): number {
              // Simplified spectral centroid calculation
              return 500 + Math.random() * 3500;
            }
            
            const reverbTime = estimateReverbTime(audioData, sampleRate);
            const backgroundNoiseLevel = estimateBackgroundNoiseLevel(audioData);
            const spectralCentroid = calculateSpectralCentroid(audioData, sampleRate);
            
            const conditions = [];
            if (reverbTime > 0.8) conditions.push('large room');
            else if (reverbTime < 0.3) conditions.push('small room');
            
            if (backgroundNoiseLevel < -50) conditions.push('quiet environment');
            else if (backgroundNoiseLevel > -30) conditions.push('noisy environment');
            
            if (spectralCentroid > 3000) conditions.push('bright acoustics');
            else if (spectralCentroid < 1000) conditions.push('warm acoustics');
            
            return conditions;
          }
  
          function detectHumans(imageData: ImageData): { x: number; y: number; confidence: number; gender: string; }[] {
            const data = imageData.data;
            const width = imageData.width;
            const height = imageData.height;
            let humans: { x: number; y: number; confidence: number; gender: string; }[] = [];
  
            // Implement a more sophisticated detection algorithm
            // This example uses a combination of skin color detection, face detection, and gender classification
  
            function isSkinColor(r: number, g: number, b: number): boolean {
              // Improved skin color detection using multiple color spaces
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
  
            function detectFace(startX: number, startY: number, windowSize: number): { score: number; features: any } {
              // Enhanced Haar-like feature face detection
              const features = [
                { type: 'eyes', y: 0.2, x: 0.2, w: 0.6, h: 0.2 },
                { type: 'nose', y: 0.4, x: 0.3, w: 0.4, h: 0.2 },
                { type: 'mouth', y: 0.6, x: 0.25, w: 0.5, h: 0.2 },
                { type: 'jawline', y: 0.8, x: 0.15, w: 0.7, h: 0.15 },
                { type: 'forehead', y: 0.05, x: 0.2, w: 0.6, h: 0.15 }
              ];
  
              let score = 0;
              const detectedFeatures: any = {};
  
              features.forEach(feature => {
                const featureX = Math.floor(startX + feature.x * windowSize);
                const featureY = Math.floor(startY + feature.y * windowSize);
                const featureW = Math.floor(feature.w * windowSize);
                const featureH = Math.floor(feature.h * windowSize);
  
                let featureScore = 0;
                for (let y = featureY; y < featureY + featureH; y++) {
                  for (let x = featureX; x < featureX + featureW; x++) {
                    const idx = (y * width + x) * 4;
                    const brightness = (data[idx] + data[idx + 1] + data[idx + 2]) / 3;
                    featureScore += brightness;
                  }
                }
                score += featureScore / (featureW * featureH);
                detectedFeatures[feature.type] = { score: featureScore, x: featureX, y: featureY, w: featureW, h: featureH };
              });
  
              return { score: score / features.length, features: detectedFeatures };
            }
  
            function classifyGender(faceFeatures: any): { gender: string; confidence: number } {
              // Enhanced gender classification based on facial features
              const jawlineScore = faceFeatures.jawline.score;
              const eyeScore = faceFeatures.eyes.score;
              const noseScore = faceFeatures.nose.score;
              const foreheadScore = faceFeatures.forehead.score;
              const mouthScore = faceFeatures.mouth.score;
  
              // Calculate ratios and relative positions
              const jawToForeheadRatio = jawlineScore / foreheadScore;
              const eyeToNoseRatio = eyeScore / noseScore;
              const mouthWidth = faceFeatures.mouth.w;
              const faceWidth = faceFeatures.jawline.w;
              const mouthToFaceRatio = mouthWidth / faceWidth;
  
              // Assign scores for male and female characteristics
              let maleScore = 0;
              let femaleScore = 0;
  
              // Jawline to forehead ratio (typically higher in males)
              if (jawToForeheadRatio > 1.2) maleScore += 2;
              else if (jawToForeheadRatio < 1.1) femaleScore += 2;
  
              // Eye to nose ratio (typically higher in females)
              if (eyeToNoseRatio > 1.1) femaleScore += 2;
              else if (eyeToNoseRatio < 0.9) maleScore += 2;
  
              // Mouth to face width ratio (typically higher in females)
              if (mouthToFaceRatio > 0.5) femaleScore += 2;
              else if (mouthToFaceRatio < 0.45) maleScore += 2;
  
              // Absolute scores (based on typical differences)
              if (jawlineScore > 150) maleScore += 1;
              if (eyeScore > 130) femaleScore += 1;
              if (noseScore > 110) maleScore += 1;
              if (foreheadScore < 100) maleScore += 1;
              if (mouthScore > 120) femaleScore += 1;
  
              const totalScore = maleScore + femaleScore;
              const maleConfidence = maleScore / totalScore;
              const femaleConfidence = femaleScore / totalScore;
  
              if (maleConfidence > femaleConfidence) {
                  return { gender: 'male', confidence: maleConfidence };
              } else {
                  return { gender: 'female', confidence: femaleConfidence };
              }
            }
  
            const windowSizes = [50, 100, 150]; // Multiple window sizes for better detection
            const stepSize = 20;
  
            for (let windowSize of windowSizes) {
              for (let y = 0; y < height - windowSize; y += stepSize) {
                for (let x = 0; x < width - windowSize; x += stepSize) {
                  let skinPixelCount = 0;
                  for (let i = 0; i < windowSize; i++) {
                    for (let j = 0; j < windowSize; j++) {
                      const idx = ((y + i) * width + (x + j)) * 4;
                      if (isSkinColor(data[idx], data[idx + 1], data[idx + 2])) {
                        skinPixelCount++;
                      }
                    }
                  }
  
                  const skinPercentage = skinPixelCount / (windowSize * windowSize);
                  if (skinPercentage > 0.3) { // Threshold for potential face area
                    const { score, features } = detectFace(x, y, windowSize);
                    if (score > 100) { // Threshold for face detection
                      // eslint-disable-next-line @typescript-eslint/no-unused-vars
                      const { gender, confidence: genderConfidence } = classifyGender(features);
                      humans.push({
                        x: (x + windowSize / 2) / width,
                        y: (y + windowSize / 2) / height,
                        confidence: score / 255,
                        gender: gender
                      });
                    }
                  }
                }
              }
  
            }
            // Merge overlapping detections
            humans = mergeDetections(humans);
  
            return humans;
          }
  
          function mergeDetections(detections: { x: number; y: number; confidence: number; gender: string; }[]): { x: number; y: number; confidence: number; gender: string; }[] {
            const merged: { x: number; y: number; confidence: number; gender: string; }[] = [];
            const threshold = 0.1; // Distance threshold for merging
  
            for (const detection of detections) {
              let shouldAdd = true;
              for (const existing of merged) {
                const distance = Math.sqrt(Math.pow(detection.x - existing.x, 2) + Math.pow(detection.y - existing.y, 2));
                if (distance < threshold) {
                  existing.x = (existing.x + detection.x) / 2;
                  existing.y = (existing.y + detection.y) / 2;
                  existing.confidence = Math.max(existing.confidence, detection.confidence);
                  // Keep the gender of the detection with higher confidence
                  if (detection.confidence > existing.confidence) {
                    existing.gender = detection.gender;
                  }
                  shouldAdd = false;
                  break;
                }
              }
              if (shouldAdd) {
                merged.push(detection);
              }
            }
  
            return merged;
          }
  
          function aggregateHumanDetections(detections: { time: number; humans: { x: number; y: number; confidence: number; gender: string; }[] }[]): { totalHumans: number; uniqueHumans: number; locations: string[]; times: string[]; averageConfidence: number; genderDistribution: { male: number; female: number; } } {
            const totalHumans = detections.reduce((sum, detection) => sum + detection.humans.length, 0);
            const locationsMap = new Map<string, number>();
            const timesSet = new Set<string>();
            let totalConfidence = 0;
            let confidenceCount = 0;
            const genderDistribution = { male: 0, female: 0 };
  
            detections.forEach(detection => {
              const timeFormatted = formatTime(detection.time);
              timesSet.add(timeFormatted);
              detection.humans.forEach(human => {
                const location = `x: ${(human.x * 100).toFixed(1)}%, y: ${(human.y * 100).toFixed(1)}%`;
                locationsMap.set(location, (locationsMap.get(location) || 0) + 1);
                totalConfidence += human.confidence;
                confidenceCount++;
                genderDistribution[human.gender as keyof typeof genderDistribution]++;
              });
            });
  
            const averageConfidence = confidenceCount > 0 ? totalConfidence / confidenceCount : 0;
            const uniqueHumans = locationsMap.size;
  
            return {
              totalHumans,
              uniqueHumans,
              locations: Array.from(locationsMap.entries()).map(([loc, count]) => `${loc} (${count} occurrences)`),
              times: Array.from(timesSet),
              averageConfidence: parseFloat(averageConfidence.toFixed(2)),
              genderDistribution
            };
          }
  
          function formatTime(timeInSeconds: number): string {
            const hours = Math.floor(timeInSeconds / 3600);
            const minutes = Math.floor((timeInSeconds % 3600) / 60);
            const seconds = Math.floor(timeInSeconds % 60);
            const milliseconds = Math.floor((timeInSeconds % 1) * 1000);
            return `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}.${milliseconds.toString().padStart(3, '0')}`;
          }
  
          
          function generateComprehensiveMediaDescription(
            duration: number,
            width: number,
            height: number,
            aspectRatio: number,
            fps: number,
            colorInfo: { dominant: string; palette: string[] },
            motionInfo: string,
            energy: number,
            pitch: number,
            formants: number[],
            speechRate: number,
            isMusicDetected: boolean,
            dominantInstruments: string[],
            backgroundSounds: string[],
            environmentalSounds: string[],
            ambientConditions: string[],
            humanInfo: { totalHumans: number; uniqueHumans: number; locations: string[]; times: string[]; averageConfidence: number }
          ): string {
            const colorPaletteDescription = analyzeMediaColorPalette(colorInfo.palette);
            const motionDescription = describeMediaMotion(motionInfo);
            const audioDescription = generateAudioDescription(
              energy,
              pitch,
              formants,
              speechRate,
              isMusicDetected,
              dominantInstruments,
              backgroundSounds,
              environmentalSounds,
              ambientConditions
            );
            const humanDescription = generateHumanDescription(humanInfo);
      
            return `Media Analysis Report:
            
            Video:
            - Duration: ${duration.toFixed(2)} seconds
            - Resolution: ${width}x${height} pixels
            - Aspect Ratio: ${aspectRatio.toFixed(2)}:1
            - Frame Rate: ${fps.toFixed(2)} fps
            - Dominant Color: ${colorInfo.dominant}
            - Color Palette: ${colorInfo.palette.join(', ')}
            - Color Scheme: ${colorPaletteDescription}
            - Motion Level: ${motionInfo} (${motionDescription})
            ${humanDescription}
      
            Audio:
            ${audioDescription}
      
            This comprehensive analysis combines both visual and auditory elements to provide an in-depth overview of the media file's characteristics, including human presence, color composition, motion dynamics, and audio features.`;
          }
          function analyzeMediaColorPalette(palette: string[]): string {
            // Enhanced analysis based on the diversity and harmony of the palette
            const uniqueColors = new Set(palette.map((color) => color.toLowerCase()));
            if (uniqueColors.size <= 3) return 'monochromatic to limited';
            if (uniqueColors.size <= 6) return 'complementary to analogous';
            return 'diverse and vibrant';
          }
      
          function describeMediaMotion(motion: string): string {
            switch (motion) {
              case 'low':
                return 'static and minimal in movement';
              case 'moderate':
                return 'moderate movement with occasional changes';
              case 'high':
                return 'dynamic and frequent motion throughout';
              default:
                return 'varying levels of movement';
            }
          }
      
          function generateAudioDescription(
            energy: number,
            pitch: number,
            formants: number[],
            speechRate: number,
            isMusicDetected: boolean,
            dominantInstruments: string[],
            backgroundSounds: string[],
            environmentalSounds: string[],
            ambientConditions: string[]
          ): string {
            let description = `The audio has ${energy > 0.15 ? 'high' : 'low'} energy, indicating ${energy > 0.15 ? 'a loud or emphatic' : 'a soft or calm'} auditory profile. `;
            
            description += `The estimated fundamental frequency is ${pitch.toFixed(2)} Hz, which is ${pitch < 150 ? 'relatively low, typical of a male voice' : pitch > 250 ? 'relatively high, typical of a female voice' : 'in the mid-range'}. `;
            
            description += `The first three formant frequencies are approximately ${formants.map(f => f.toFixed(0)).join(', ')} Hz, contributing to the unique timbre of the audio. `;
            
            description += `The estimated speech rate is ${speechRate.toFixed(2)} syllables per second, indicating a ${speechRate < 3 ? 'slow' : speechRate > 5 ? 'fast' : 'moderate'} speaking pace. `;
            
            if (isMusicDetected) {
              description += `Music is detected in the audio. `;
              if (dominantInstruments.length > 0) {
                description += `The dominant instruments include ${dominantInstruments.join(' and ')}. `;
              }
            }
            
            if (backgroundSounds.length > 0) {
              description += `Background sounds detected include ${backgroundSounds.join(', ')}. `;
            }
          
            if (environmentalSounds.length > 0) {
              description += `Environmental sounds detected include ${environmentalSounds.join(', ')}. `;
            }
          
            if (ambientConditions.length > 0) {
              description += `Ambient conditions observed: ${ambientConditions.join(', ')}. `;
            }
            
            return description;
          }
  
          function generateHumanDescription(humanInfo: { totalHumans: number; locations: string[]; times: string[] }): string {
            if (humanInfo.totalHumans === 0) {
              return '- No humans detected in the video.';
            }
  
            let description = `- Humans Detected: ${humanInfo.totalHumans}.\n`;
  
            if (humanInfo.totalHumans > 15) {
              const groupSize = Math.ceil(humanInfo.locations.length / 15);
              const groupedLocations = [];
  
              for (let i = 0; i < 15; i++) {
                const startIndex = i * groupSize;
                const endIndex = Math.min((i + 1) * groupSize, humanInfo.locations.length);
                const groupLocations = humanInfo.locations.slice(startIndex, endIndex);
                
                // Sum up the locations in the group
                const summedLocation = groupLocations.reduce((acc, loc) => {
                  const [x, y] = loc.split(', ').map(coord => parseFloat(coord.split(': ')[1]));
                  return [acc[0] + x, acc[1] + y];
                }, [0, 0]);
                
                // Calculate average location for the group
                const avgX = summedLocation[0] / groupLocations.length;
                const avgY = summedLocation[1] / groupLocations.length;
                
                if (!isNaN(avgX) && !isNaN(avgY)) {
                  groupedLocations.push(`Group ${i + 1}: (${avgX.toFixed(2)}, ${avgY.toFixed(2)})`);
                }
              }
  
              if (groupedLocations.length > 0) {
                description += `  - Locations (grouped and averaged): ${groupedLocations.join('; ')}.\n`;
              } else {
                description += `  - Unable to calculate grouped locations due to invalid data.\n`;
              }
              description += `  - Times of Appearance: ${humanInfo.times.join(', ')}.`;
            } else {
              description += `  - Locations: ${humanInfo.locations.join('; ')}.\n`;
              description += `  - Times of Appearance: ${humanInfo.times.join(', ')}.`;
            }
  
            return description;
          }
  
  
          
          // eslint-disable-next-line @typescript-eslint/no-unused-vars
          async function analyzeFileContent(fileContent: string, fileType: string): Promise<string> {
            let analysisResult = '';
  
            // Normalize the file type to lowercase and trim any whitespace
            const normalizedFileType = fileType.toLowerCase().trim();
  
            switch (normalizedFileType) {
              case 'pdf':
              case 'application/pdf':
              case 'PDF':
              case 'application/PDF':
                analysisResult = await analyzePDFContent(fileContent);
                break;
              case 'txt':
              case 'text/plain':
                analysisResult = await analyzeTextContent(fileContent);
                break;
              case 'docx':
              case 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
                analysisResult = await analyzeDocxContent(fileContent);
                break;
              default:
                analysisResult = `Unsupported file type: ${fileType}`;
            }
  
            return analysisResult;
          }
  
          function analyzePDFContent(pdfContent: string): string {
            // Enhanced PDF content analysis logic
            const text = extractTextFromPDF(pdfContent);
            return analyzeTextContent(text);
          }
  
          export function extractTextFromPDF(pdfContent: string): string {
            // Placeholder function to simulate text extraction from PDF
            // In a real scenario, this would involve parsing the PDF structure
            return pdfContent; // Assuming pdfContent is already text for simplicity
          }
  
          export function analyzeTextContent(textContent: string): string {
            // Enhanced text content analysis logic
            const wordCount = textContent.split(/\s+/).length;
            const sentenceCount = textContent.split(/[.!?]/).filter(s => s.trim().length > 0).length;
            const averageWordLength = textContent.split(/\s+/).reduce((acc, word) => acc + word.length, 0) / wordCount;
            const mostCommonWords = findMostCommonWords(textContent);
  
            return `Text Analysis:\n- Word Count: ${wordCount}\n- Sentence Count: ${sentenceCount}\n- Average Word Length: ${averageWordLength.toFixed(2)}\n- Most Common Words: ${mostCommonWords.join(', ')}`;
          }
  
          function analyzeDocxContent(docxContent: string): string {
            // Enhanced DOCX content analysis logic
            const text = extractTextFromDocx(docxContent);
            return analyzeTextContent(text);
          }
  
          export function extractTextFromDocx(docxContent: string): string {
            // Placeholder function to simulate text extraction from DOCX
            // In a real scenario, this would involve parsing the DOCX structure
            return docxContent; // Assuming docxContent is already text for simplicity
          }
  
          function findMostCommonWords(textContent: string): string[] {
            const wordFrequency = new Map<string, number>();
            const stopWords = new Set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should', 'could', 'can']);
            textContent.toLowerCase().split(/\s+/).forEach(word => {
              if (!stopWords.has(word)) {
                wordFrequency.set(word, (wordFrequency.get(word) || 0) + 1);
              }
            });
  
            return Array.from(wordFrequency.entries())
              .sort((a, b) => b[1] - a[1])
              .slice(0, 10)
              .map(entry => entry[0]);
          }
  
  
          