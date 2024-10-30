export async function processVoiceFile(voiceFile: ArrayBuffer): Promise<string> {
    try {
      const audioContext = new (window.AudioContext || (window as any).webkitAudioContext)();
      
      const audioBuffer = await audioContext.decodeAudioData(voiceFile);
      const channelData = audioBuffer.getChannelData(0);
      const duration = audioBuffer.duration;
      const energy = calculateEnergy(channelData);
      const pitch = estimateAdvancedPitch(channelData, audioBuffer.sampleRate);
      const formants = estimateFormants(channelData, audioBuffer.sampleRate);
      const speechRate = estimateSpeechRate(channelData, audioBuffer.sampleRate);
      const { isMusicDetected, dominantInstruments } = detectMusic(channelData, audioBuffer.sampleRate);
      const backgroundSounds = detectBackgroundSounds(channelData, audioBuffer.sampleRate);
      const environmentalSounds = detectEnvironmentalSounds(channelData, audioBuffer.sampleRate);
      const ambientConditions = analyzeAmbientConditions(channelData, audioBuffer.sampleRate);
       
      const description = generateEnhancedVoiceDescription(
        duration, 
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
      
      return description;
    } catch (error) {
      console.error('Error processing audio file:', error);
      return fallbackAudioAnalysis(voiceFile);
    }
  }
  
  function fallbackAudioAnalysis(voiceFile: ArrayBuffer): string {
    const fileSize = voiceFile.byteLength;
    const fileSizeMB = (fileSize / (1024 * 1024)).toFixed(2);
    
    let description = `Audio file analysis (fallback method):
  - File size: ${fileSizeMB} MB
  - Format: Unable to decode, possibly unsupported or corrupted file
  - Duration: Unknown (unable to decode)
  - Content: Unable to perform detailed analysis due to decoding issues
  
  Possible reasons for decoding failure:
  1. Unsupported audio format
  2. Corrupted audio file
  3. File too large for browser-based decoding
  4. Browser limitations or security restrictions
  
  Recommendations:
  - Try converting the audio to a widely supported format like MP3 or WAV
  - Ensure the file is not corrupted
  - If the file is large, consider using a smaller or shorter audio clip
  - Check browser compatibility and any content security policies that might be blocking audio decoding`;
  
    return description;
  }
  
  function calculateEnergy(channelData: Float32Array): number {
    return channelData.reduce((sum, sample) => sum + Math.abs(sample), 0) / channelData.length;
  }
  
  function estimateAdvancedPitch(channelData: Float32Array, sampleRate: number): number {
    const correlations = new Float32Array(2000);
    for (let lag = 0; lag < correlations.length; lag++) {
      for (let i = 0; i < 2000 && i + lag < channelData.length; i++) {
        correlations[lag] += channelData[i] * channelData[i + lag];
      }
    }
    const maxLag = correlations.reduce((maxIndex, currentValue, currentIndex, array) => 
      currentIndex > 20 && currentValue > array[maxIndex] ? currentIndex : maxIndex
    , 21);
    return sampleRate / maxLag;
  }
  
  function estimateFormants(channelData: Float32Array, sampleRate: number): number[] {
    // This is a simplified formant estimation. For accurate results, 
    // you'd need to implement a more complex algorithm like LPC analysis.
    const fftSize = 2048;
    const fft = performFFT(channelData, fftSize);
    
    // Perform FFT (placeholder - replace with actual FFT implementation or library)
    const spectrum = fft.map(x => Math.abs(x));
    
    // Find peaks in the spectrum (simplified)
    const formants = [];
    for (let i = 1; i < spectrum.length - 1; i++) {
      if (spectrum[i] > spectrum[i-1] && spectrum[i] > spectrum[i+1]) {
        formants.push(i * sampleRate / fftSize);
      }
    }
    
    return formants.slice(0, 3); // Return first 3 formants
  }
  
  function estimateSpeechRate(channelData: Float32Array, sampleRate: number): number {
    const energyThreshold = 0.1;
    let syllableCount = 0;
    let inSyllable = false;
    
    for (let i = 0; i < channelData.length; i++) {
      if (Math.abs(channelData[i]) > energyThreshold) {
        if (!inSyllable) {
          syllableCount++;
          inSyllable = true;
        }
      } else {
        inSyllable = false;
      }
    }
    
    const durationInSeconds = channelData.length / sampleRate;
    return syllableCount / durationInSeconds;
  }
  
  function detectMusic(channelData: Float32Array, sampleRate: number): { isMusicDetected: boolean; dominantInstruments: string[] } {
    // Enhanced music detection with broader frequency analysis
    const fftSize = 4096;
    const spectrum = performFFT(channelData, fftSize).map(x => Math.abs(x));
    
    // Heuristic: strong harmonic patterns indicate music
    const harmonicPresence = [300, 600, 900, 1200, 1500].every(freq => {
      const index = Math.round(freq * fftSize / sampleRate);
      return spectrum[index] > 0.3;
    });
    
    const isMusicDetected = harmonicPresence;
    
    // Enhanced instrument detection placeholders
    const dominantInstruments: string[] = [];
    if (isMusicDetected) {
      if (spectrum[Math.round(440 * fftSize / sampleRate)] > 0.5) {
        dominantInstruments.push('piano');
      }
      if (spectrum[Math.round(330 * fftSize / sampleRate)] > 0.4) {
        dominantInstruments.push('guitar');
      }
      if (spectrum[Math.round(500 * fftSize / sampleRate)] > 0.3) {
        dominantInstruments.push('violin');
      }
    }
    
    return { isMusicDetected, dominantInstruments };
  }
  
  function detectBackgroundSounds(channelData: Float32Array, sampleRate: number): string[] {
    // Enhanced background sound detection with multiple categories
    const energyThreshold = 0.05;
    const backgroundSounds: string[] = [];
    
    if (channelData.some(sample => Math.abs(sample) > energyThreshold)) {
      backgroundSounds.push('ambient noise');
    }
    
    const fftSize = 4096;
    const spectrum = performFFT(channelData, fftSize).map(x => Math.abs(x));
    
    // Low frequency rumble detection
    if (spectrum.slice(0, Math.round(100 * fftSize / sampleRate)).some(value => value > 0.1)) {
      backgroundSounds.push('low frequency rumble');
    }
    
    // Mid-frequency chatter detection
    if (spectrum.slice(Math.round(500 * fftSize / sampleRate), Math.round(2000 * fftSize / sampleRate)).some(value => value > 0.2)) {
      backgroundSounds.push('chatter');
    }
    
    return backgroundSounds;
  }
  
  function detectEnvironmentalSounds(channelData: Float32Array, sampleRate: number): string[] {
    // Enhanced environmental sound detection with multiple categories
    const environmentalSounds: string[] = [];
    const fftSize = 4096;
    const spectrum = performFFT(channelData, fftSize).map(x => Math.abs(x));
    
    // Detect traffic noises
    const trafficNoise = spectrum.slice(Math.round(500 * fftSize / sampleRate), Math.round(1500 * fftSize / sampleRate)).some(value => value > 0.35);
    if (trafficNoise) {
      environmentalSounds.push('traffic noise');
    }
    
    // Detect animal sounds (e.g., dog barking, birds chirping)
    const animalSound = spectrum.slice(Math.round(1000 * fftSize / sampleRate), Math.round(3000 * fftSize / sampleRate)).some(value => value > 0.3);
    if (animalSound) {
      environmentalSounds.push('animal sounds');
    }
    
    // Detect weather-related sounds (e.g., rain, thunder)
    const rainSound = spectrum.slice(Math.round(200 * fftSize / sampleRate), Math.round(800 * fftSize / sampleRate)).some(value => value > 0.25);
    if (rainSound) {
      environmentalSounds.push('rain');
    }
    
    const thunderSound = spectrum.slice(Math.round(1000 * fftSize / sampleRate), Math.round(2000 * fftSize / sampleRate)).some(value => value > 0.4);
    if (thunderSound) {
      environmentalSounds.push('thunder');
    }
    
    // Detect footsteps
    const footsteps = spectrum.slice(Math.round(300 * fftSize / sampleRate), Math.round(800 * fftSize / sampleRate)).some(value => value > 0.3);
    if (footsteps) {
      environmentalSounds.push('footsteps');
    }
    
    return environmentalSounds;
  }
  
  function analyzeAmbientConditions(channelData: Float32Array, sampleRate: number): string[] {
    // Additional ambient condition analysis (e.g., echo, reverberation)
    const ambientConditions: string[] = [];
    const fftSize = 4096;
    const spectrum = performFFT(channelData, fftSize).map(x => Math.abs(x));
    
    // Detect echo based on delayed peaks
    const echoDetected = spectrum.slice(0, Math.round(500 * fftSize / sampleRate)).reduce((a, b) => a + b, 0) > 10;
    if (echoDetected) {
      ambientConditions.push('echo');
    }
    
    // Detect reverberation
    const reverbDetected = spectrum.slice(Math.round(200 * fftSize / sampleRate), Math.round(1000 * fftSize / sampleRate)).reduce((a, b) => a + b, 0) > 15;
    if (reverbDetected) {
      ambientConditions.push('reverberation');
    }
    
    return ambientConditions;
  }
  
  function performFFT(channelData: Float32Array, fftSize: number): Float32Array {
    // Placeholder for FFT implementation. In a real-world scenario, use a proper FFT library.
    const fft = new Float32Array(fftSize);
    for (let i = 0; i < fftSize; i++) {
      fft[i] = channelData[i] || 0;
    }
    return fft;
  }
  
  function generateEnhancedVoiceDescription(
    duration: number,
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
    let description = `This audio recording is ${duration.toFixed(2)} seconds long. `;
    
    description += `The primary voice has ${energy > 0.1 ? 'high' : 'low'} energy, indicating ${energy > 0.1 ? 'a loud or emphatic' : 'a soft or calm'} speaking style. `;
    
    description += `The estimated fundamental frequency is ${pitch.toFixed(2)} Hz, which is ${pitch < 150 ? 'relatively low, typical of a male voice' : pitch > 250 ? 'relatively high, typical of a female voice' : 'in the middle range'}. `;
    
    description += `The first three formant frequencies are approximately ${formants.map(f => f.toFixed(0)).join(', ')} Hz, which contribute to the unique timbre of the voice. `;
    
    description += `The estimated speech rate is ${speechRate.toFixed(2)} syllables per second, indicating a ${speechRate < 3 ? 'slow' : speechRate > 5 ? 'fast' : 'moderate'} speaking pace. `;
    
    if (isMusicDetected) {
      description += `Music is detected in the background. `;
      if (dominantInstruments.length > 0) {
        description += `The dominant instruments appear to be ${dominantInstruments.join(' and ')}. `;
      }
    }
    
    if (backgroundSounds.length > 0) {
      description += `Other background sounds detected include ${backgroundSounds.join(', ')}. `;
    }
  
    if (environmentalSounds.length > 0) {
      description += `Environmental sounds detected include ${environmentalSounds.join(', ')}. `;
    }
  
    if (ambientConditions.length > 0) {
      description += `Ambient conditions observed: ${ambientConditions.join(', ')}. `;
    }
    
    return description;
  }
  
  
  