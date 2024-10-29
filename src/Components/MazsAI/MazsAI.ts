/* eslint-disable no-eval */
/* eslint-disable no-template-curly-in-string */
import { processImageFile } from './ImageProcessor';
import { processMediaFile } from './VideoModel';
import {extractTextFromPDF} from './VideoModel'
import {extractTextFromDocx} from './VideoModel'
import {processVoiceFile} from './AudioModel'
interface Intent {
    patterns: string[];
    responses: string[];
  }
  
  class MultilayerPerceptron {
    private layers: number[];
    private weights: number[][][];
    private biases: number[][];
    private activations: ((x: number) => number)[];
    private activationDerivatives: ((x: number) => number)[];
    private optimizer: Optimizer;
    private learningRate: number;
    private batchSize: number; 
    private epochs: number;
    private taskHeads: { [task: string]: { weights: number[][], biases: number[] } };
    private tasks: string[];
  
   constructor(
      layers: number[],
      activations: string[] = [],
      learningRate: number = 0.05,
      batchSize: number = 64,
      epochs: number = 1,
      tasks: string[] = []
    ) {
      this.layers = layers;
      this.weights = [];
      this.biases = [];
      this.activations = [];
      this.activationDerivatives = [];
      this.learningRate = learningRate;
      this.batchSize = batchSize;
      this.epochs = epochs;
      this.optimizer = new AdamOptimizer(learningRate);
      this.tasks = tasks;
      this.taskHeads = {};
  
     for (let i = 1; i < layers.length; i++) {
        this.weights.push(
          Array(layers[i])
            .fill(0)
            .map(() =>
              Array(layers[i - 1])
                .fill(0)
                .map(() => this.initializeWeight(layers[i - 1], layers[i]))
            )
        );
        this.biases.push(
          Array(layers[i])
            .fill(0)
            .map(() => this.initializeWeight(1, layers[i]))
        );
  
        const activation = activations[i - 1] || 'relu';
        this.activations.push(this.getActivationFunction(activation));
        this.activationDerivatives.push(this.getActivationDerivative(activation));
      }
  
      // Initialize task-specific heads
      this.initializeTaskHeads();
    }
    private initializeTaskHeads() {
      this.tasks.forEach((task) => {
        const outputSize = this.getOutputSizeForTask(task);
        this.taskHeads[task] = {
          weights: Array(outputSize)
            .fill(0)
            .map(() =>
              Array(this.layers[this.layers.length - 1])
                .fill(0)
                .map(() => this.initializeWeight(this.layers[this.layers.length - 1], 1))
            ),
          biases: Array(outputSize)
            .fill(0)
            .map(() => this.initializeWeight(1, 1)),
        };
      });
    }
    private getOutputSizeForTask(task: string): number {
      // Define output size based on task type
      const taskOutputSizes: { [key: string]: number } = {
        classification: 1, // Binary classification
        regression: 1,     // Single value regression
        // Extend as needed for other tasks
      };
      return taskOutputSizes[task] || 1;
    }
    private initializeWeight(inputSize: number, outputSize: number): number {
      // Enhanced Xavier/Glorot initialization with He initialization for ReLU
      const isReLU = this.activations.some(act => act.name.includes('relu'));
      const fanIn = inputSize;
      const fanOut = outputSize;
      
      if (isReLU) {
        // He initialization for ReLU activation
        const stdDev = Math.sqrt(2 / fanIn);
        return this.gaussianRandom(0, stdDev);
      } else {
        // Xavier/Glorot initialization for other activations
        const limit = Math.sqrt(6 / (fanIn + fanOut));
        return Math.random() * 2 * limit - limit;
      }
    }
  
    private gaussianRandom(mean: number, stdDev: number): number {
      const u = 1 - Math.random(); // Converting [0,1) to (0,1]
      const v = Math.random();
      const z = Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
      return z * stdDev + mean;
    }
  
    private getActivationFunction(name: string): (x: number) => number {
      switch (name) {
        case 'sigmoid': return (x: number) => 1 / (1 + Math.exp(-x));
        case 'relu': return (x: number) => Math.max(0, x);
        case 'tanh': return (x: number) => Math.tanh(x);
        case 'leaky_relu': return (x: number) => x > 0 ? x : 0.01 * x;
        default: return (x: number) => Math.max(0, x); // default to relu
      }
    }
  
    private getActivationDerivative(name: string): (x: number) => number {
      const derivatives: { [key: string]: (x: number) => number } = {
        sigmoid: (x: number) => {
          const s = 1 / (1 + Math.exp(-x));
          return s * (1 - s);
        },
        relu: (x: number) => x > 0 ? 1 : 0,
        tanh: (x: number) => 1 - Math.pow(Math.tanh(x), 2),
        leaky_relu: (x: number) => x > 0 ? 1 : 0.01,
        elu: (x: number) => x >= 0 ? 1 : Math.exp(x),
        swish: (x: number) => {
          const sigmoid = 1 / (1 + Math.exp(-x));
          return sigmoid + x * sigmoid * (1 - sigmoid);
        },
      };
  
      return derivatives[name] || ((x: number) => x > 0 ? 1 : 0); // default to relu derivative
    }
  
    // Add a method for batch normalization
    private batchNormalize(layer: number[]): number[] {
      const mean = layer.reduce((sum, val) => sum + val, 0) / layer.length;
      const variance = layer.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / layer.length;
      return layer.map(val => (val - mean) / Math.sqrt(variance + 1e-8));
    }
  
    // Add a method for dropout regularization
    private applyDropout(layer: number[], rate: number): number[] {
      return layer.map(val => Math.random() > rate ? val / (1 - rate) : 0);
    }
  
    // Modify the predict method to include batch normalization
    predict(input: number[]): number[] {
      let activation = this.batchNormalize(input);
      for (let i = 0; i < this.weights.length; i++) {
        const currentActivation = activation;
        let newActivation = [];
        for (let j = 0; j < this.weights[i].length; j++) {
          const sum = this.weights[i][j].reduce((sum, weight, k) => sum + weight * currentActivation[k], 0) + this.biases[i][j];
          const activatedValue = this.activations[i](sum);
          newActivation.push(this.applyDropout([activatedValue], 0.1)[0]); // Apply lightweight dropout
        }
        activation = this.batchNormalize(newActivation);
        
        // Add skip connection if not the last layer
        if (i < this.weights.length - 1 && activation.length === currentActivation.length) {
          activation = activation.map((val, idx) => val + currentActivation[idx]);
        }
      }
      
      // Apply softmax to get probabilities
      const expValues = activation.map(Math.exp);
      const sumExpValues = expValues.reduce((a, b) => a + b, 0);
      const probabilities = expValues.map(value => value / sumExpValues);
      
      // Use dynamic thresholding
      const meanProb = probabilities.reduce((a, b) => a + b, 0) / probabilities.length;
      const threshold = meanProb * 0.8; // Adjust this factor as needed
      
      return probabilities.map((prob) => (prob >= threshold ? 1 : 0));
    }
  
    // Modify the train method to include dropout and advanced optimizers
    train(input: number[], target: number[], learningRate: number = 0.1, momentum: number = 0.9, dropoutRate: number = 0.2) {
      // Forward pass with lightweight enhancements
      let activations: number[][] = [this.batchNormalize(input)];
      let weightedSums: number[][] = [];
      let dropoutMasks: boolean[][] = [];
  
      for (let i = 0; i < this.weights.length; i++) {
        const [newActivation, newWeightedSum] = this.forwardLayer(activations[i], this.weights[i], this.biases[i], this.activations[i]);
        weightedSums.push(newWeightedSum);
        
        // Apply dropout
        const mask = this.generateDropoutMask(newActivation.length, dropoutRate);
        dropoutMasks.push(mask);
        const droppedActivation = newActivation.map((a, idx) => mask[idx] ? a / (1 - dropoutRate) : 0);
        
        // Apply batch normalization and add to activations
        activations.push(this.batchNormalize(droppedActivation));
      }
  
      // Backward pass with improvements
      let deltas = this.calculateOutputDeltas(activations[activations.length - 1], target, weightedSums[weightedSums.length - 1]);
  
      for (let i = this.weights.length - 1; i > 0; i--) {
        deltas = [this.calculateHiddenDeltas(deltas[0], this.weights[i], weightedSums[i-1], this.activationDerivatives[i-1]), ...deltas];
      }
  
      // Update weights and biases with advanced optimizer and regularization
      this.optimizer.update(this.weights, this.biases, activations, deltas, learningRate, momentum, dropoutRate);
      this.applyWeightDecay(learningRate * 0.0001); // Lightweight L2 regularization
    }
  
    private forwardLayer(inputs: number[], weights: number[][], biases: number[], activation: (x: number) => number): [number[], number[]] {
      const weightedSum = weights.map((neuronWeights, j) => 
        neuronWeights.reduce((sum, weight, k) => sum + weight * inputs[k], 0) + biases[j]
      );
      return [weightedSum.map(activation), weightedSum];
    }
  
    private generateDropoutMask(size: number, rate: number): boolean[] {
      return Array(size).fill(0).map(() => Math.random() > rate);
    }
  
    private calculateOutputDeltas(outputs: number[], targets: number[], weightedSums: number[]): number[][] {
      return [outputs.map((output, i) => 
        (output - targets[i]) * this.activationDerivatives[this.activationDerivatives.length - 1](weightedSums[i])
      )];
    }
  
    private calculateHiddenDeltas(nextDeltas: number[], weights: number[][], weightedSums: number[], activationDerivative: (x: number) => number): number[] {
      return weightedSums.map((sum, j) => {
        const error = weights.reduce((acc, neuronWeights, k) => acc + neuronWeights[j] * nextDeltas[k], 0);
        return error * activationDerivative(sum);
      });
    }
  
  
    private applyWeightDecay(decayRate: number) {
      this.weights = this.weights.map(layer => 
        layer.map(neuron => 
          neuron.map(weight => weight * (1 - decayRate))
        )
      );
    }
  
    // Add a method for L2 regularization
    private applyL2Regularization(weights: number[][][], lambda: number, learningRate: number): number[][][] {
      return weights.map(layer => 
        layer.map(neuron => 
          neuron.map(weight => weight * (1 - lambda * learningRate))
        )
      );
    }
    
    // Modify the batchTrain method to include L2 regularization
    batchTrain(inputs: number[][], targets: number[][], learningRate: number = 0.1, batchSize: number = 64, lambda: number = 0.01) {
      for (let i = 0; i < inputs.length; i += batchSize) {
        const batchInputs = inputs.slice(i, i + batchSize);
        const batchTargets = targets.slice(i, i + batchSize);
        
        let gradients = this.weights.map(layer => layer.map(neuron => neuron.map(() => 0)));
        let biasGradients = this.biases.map(layer => layer.map(() => 0));
  
        for (let j = 0; j < batchInputs.length; j++) {
          const [deltaGradients, deltaBiasGradients] = this.backpropagate(batchInputs[j], batchTargets[j]);
          
          gradients = gradients.map((layer, l) => 
            layer.map((neuron, n) => 
              neuron.map((grad, w) => grad + deltaGradients[l][n][w])
            )
          );
          
          biasGradients = biasGradients.map((layer, l) => 
            layer.map((bias, n) => bias + deltaBiasGradients[l][n])
          );
        }
  
        // Update weights and biases with averaged gradients
        const batchLearningRate = learningRate / batchInputs.length;
        this.weights = this.weights.map((layer, l) => 
          layer.map((neuron, n) => 
            neuron.map((weight, w) => weight - batchLearningRate * gradients[l][n][w])
          )
        );
        
        this.biases = this.biases.map((layer, l) => 
          layer.map((bias, n) => bias - batchLearningRate * biasGradients[l][n])
        );
      }
  
      this.weights = this.applyL2Regularization(this.weights, lambda, learningRate);
    }
  
    // Helper method for backpropagation
    private backpropagate(input: number[], target: number[]): [number[][][], number[][]] {
      // Forward pass
      let activations: number[][] = [input];
      let weightedSums: number[][] = [];
  
      for (let i = 0; i < this.weights.length; i++) {
        let newActivation: number[] = [];
        let newWeightedSum: number[] = [];
        for (let j = 0; j < this.weights[i].length; j++) {
          const sum = this.weights[i][j].reduce((sum, weight, k) => sum + weight * activations[i][k], 0) + this.biases[i][j];
          newWeightedSum.push(sum);
          newActivation.push(this.activations[i](sum));
        }
        weightedSums.push(newWeightedSum);
        activations.push(newActivation);
      }
  
      // Backward pass
      let deltas = [activations[activations.length - 1].map((output, i) => 
        (output - target[i]) * this.activationDerivatives[this.activationDerivatives.length - 1](weightedSums[weightedSums.length - 1][i])
      )];
  
      for (let i = this.weights.length - 1; i > 0; i--) {
        let layerDelta = [];
        for (let j = 0; j < this.weights[i-1].length; j++) {
          const error = this.weights[i].reduce((sum, neuronWeights, k) => sum + neuronWeights[j] * deltas[0][k], 0);
          layerDelta.push(error * this.activationDerivatives[i-1](weightedSums[i-1][j]));
        }
        deltas.unshift(layerDelta);
      }
  
      // Calculate gradients
      let gradients = this.weights.map((layer, i) => 
        layer.map((neuron, j) => 
          neuron.map((_, k) => deltas[i][j] * activations[i][k])
        )
      );
  
      let biasGradients = deltas;
  
      return [gradients, biasGradients];
    }
  
    setLearningRate(newLearningRate: number) {
      this.learningRate = newLearningRate;
    }
  }
  
  // Optimizer classes
  class Optimizer {
    update(weights: number[][][], biases: number[][], activations: number[][], deltas: number[][], learningRate: number, momentum: number, dropoutRate: number) {
      throw new Error("Method 'update()' must be implemented.");
    }
  }
  interface Intent {
    patterns: string[];
    responses: string[];
  }
  
  class AdamOptimizer extends Optimizer {
    private beta1: number;
    private beta2: number;
    private epsilon: number;
    private m: number[][][];
    private v: number[][][];
    private t: number;
  
    constructor(learningRate: number, beta1: number = 0.9, beta2: number = 0.999, epsilon: number = 1e-8) {
      super();
      this.beta1 = beta1;
      this.beta2 = beta2;
      this.epsilon = epsilon;
      this.m = [];
      this.v = [];
      this.t = 0;
    }
  
    update(weights: number[][][], biases: number[][], activations: number[][], deltas: number[][], learningRate: number, momentum: number, dropoutRate: number) {
      this.t += 1;
      if (this.m.length === 0) {
        this.m = weights.map(layer => layer.map(neuron => neuron.map(() => 0)));
        this.v = weights.map(layer => layer.map(neuron => neuron.map(() => 0)));
      }
  
      for (let i = 0; i < weights.length; i++) {
        activations[i] = activations[i].map(val => Math.random() > dropoutRate ? val / (1 - dropoutRate) : 0);
        for (let j = 0; j < weights[i].length; j++) {
          for (let k = 0; k < weights[i][j].length; k++) {
            const grad = deltas[i][j] * activations[i][k];
            this.m[i][j][k] = this.beta1 * this.m[i][j][k] + (1 - this.beta1) * grad;
            this.v[i][j][k] = this.beta2 * this.v[i][j][k] + (1 - this.beta2) * grad * grad;
  
            const mHat = this.m[i][j][k] / (1 - Math.pow(this.beta1, this.t));
            const vHat = this.v[i][j][k] / (1 - Math.pow(this.beta2, this.t));
  
            weights[i][j][k] -= learningRate * mHat / (Math.sqrt(vHat) + this.epsilon);
          }
          biases[i][j] -= learningRate * deltas[i][j];
        }
      }
    }
  }
  class NaturalLanguageProcessor {
    private vocabulary: Set<string>;
    private wordFrequency: Map<string, number>;
    private bigramFrequency: Map<string, Map<string, number>>;
    private trigramFrequency: Map<string, Map<string, Map<string, number>>>;
    private wordVectors: Map<string, number[]>;
    private idf: Map<string, number>;
    private documents: string[];
    private contextMemory: string[];
    private sentimentLexicon: Map<string, number>;
    public knowledgeBase: Map<string, string>;
    private aiResponses: Map<string, string[]>;
    private maxContextLength: number = 5;
    private learningMemory: Map<string, { response: string, feedback: number }>;
    private feedbackThreshold: number = 0.7;
    private meaningSpace: Map<string, number[]>;
    private encoder: MultilayerPerceptron;
    private decoder: MultilayerPerceptron;
    public gan: GAN;
    public rlAgent: RLAgent;
    private conversationContext: string = '';
    private contextWindow: string[] = [];
    private maxContextWindowSize: number = 10;
    private topicKeywords: Set<string> = new Set();
    private wordProbabilities: Map<string, Map<string, number>>;
    private conversationHistory: { role: 'user' | 'ai', content: string }[] = [];
    private sentimentModel: AdvancedSentimentModel;
    private entityRecognitionModel: EntityRecognitionModel;
    private topicModel: TopicModel;
    private ngramFrequency: Map<string, number>;
    private markovChain: Map<string, Map<string, number>>;
    intents: Intent[]; // Properly typed
  
    constructor(trainingData?: number[][]) {
      this.vocabulary = new Set();
      this.wordFrequency = new Map();
      this.bigramFrequency = new Map();
      this.trigramFrequency = new Map();
      this.wordVectors = new Map();
      this.idf = new Map();
      this.documents = [];
      this.contextMemory = [];
      this.learningMemory = new Map();
      this.meaningSpace = new Map();
      this.encoder = new MultilayerPerceptron([100, 32, 64, 32, 100], ['relu', 'relu', 'relu', 'sigmoid']);
      this.decoder = new MultilayerPerceptron([100, 32, 64, 32, 100], ['relu', 'relu', 'relu', 'sigmoid']);
      this.gan = new GAN(trainingData || this.generateDummyData());
      this.rlAgent = new RLAgent();
      this.wordProbabilities = new Map();
      this.ngramFrequency = new Map();
      this.markovChain = new Map();
      
      // Expand sentiment lexicon
      this.sentimentLexicon = new Map([
        ['good', 1], ['great', 2], ['excellent', 2], ['amazing', 2], ['wonderful', 2],
        ['bad', -1], ['terrible', -2], ['awful', -2], ['horrible', -2], ['disappointing', -1],
        ['happy', 1], ['sad', -1], ['angry', -2], ['pleased', 1], ['unhappy', -1],
        ['love', 2], ['hate', -2], ['like', 1], ['dislike', -1], ['adore', 2],
        ['excited', 2], ['bored', -1], ['interested', 1], ['fascinating', 2], ['dull', -1],
        ['brilliant', 2], ['stupid', -2], ['smart', 1], ['clever', 1], ['foolish', -1],
      ]);
  
      // Expand knowledge base
      this.knowledgeBase = new Map([
        ['ai', 'Artificial intelligence (AI) refers to the simulation of human intelligence in machines.'],
        ['artificial intelligence', 'AI is the simulation of human intelligence in machines.'],
        ['machine learning', 'ML is a subset of AI that enables systems to learn and improve from experience.'],
        ['deep learning', 'Deep learning is a subset of ML using neural networks with multiple layers.'],
        ['natural language processing', 'NLP is a branch of AI that helps computers understand and interpret human language.'],
        ['neural networks', 'Neural networks are a type of machine learning model inspired by the structure of the human brain.'],
        ['gpt', 'GPT stands for Generative Pre-trained Transformer. It is a type of AI model used for natural language processing tasks.'],
        ['time', 'The current date is ' + new Date().toLocaleDateString()],
        ['who are you', 'I am Mazs AI, your virtual assistant. How can I help you today?'],
        ['poem', 'Sure, here is a poem I wrote: ' + this.generatePoem()],
        ['nlp', 'NLP is a technology that allows machines to understand and process human language.'],
        ['NLP', 'NLP is short for Natural Language Processing, which is a technology that allows machines to understand and process human language.'],
        ['Natural Language Processing', 'NLP is a technology that allows machines to understand and process human language.'],
              ]);
  
      // Add basic AI responses
      this.aiResponses = new Map([
        ['greeting', [
          "Hello! How can I assist you today? Experience the latest Mazs AI model at [MazsAI]",
          "Hi there! What would you like to know? Try out our newest features at [MazsAI]",
          "Greetings! I'm here to help. What's on your mind? Check out our latest model at [MazsAI]",
          "Welcome! How may I be of service? Experience the cutting-edge Mazs AI at [MazsAI]",
          "Good day! What can I help you with? Explore our newest capabilities at [MazsAI]"
        ]],
        ['farewell', [
          "Goodbye! Have a great day! Don't forget to try our latest model at [MazsAI]",
          "Take care! Feel free to return if you have more questions. Experience our newest features at [MazsAI]",
          "Farewell! It was a pleasure assisting you. Explore more with our latest model at [MazsAI]",
          "Until next time! Stay curious and check out our latest AI advancements at [MazsAI]",
          "Bye for now! Remember, I'm always here if you need information. Try our newest model at [MazsAI]"
        ]],
        ['thanks', [
          "You're welcome! I'm glad I could help. Experience even more with our latest model at [MazsAI]",
          "It's my pleasure to assist you! Discover our newest features at [MazsAI]",
          "I'm happy I could be of help. Is there anything else you'd like to know? Try our latest model at [MazsAI]",
          "Anytime! Don't hesitate to ask if you have more questions. Explore our cutting-edge AI at [MazsAI]",
          "I'm here to help! Feel free to ask about any other topics you're curious about. Check out our newest capabilities at [MazsAI]"
        ]],
        ['confusion', [
          "I apologize, but I'm not sure I understand. Could you please rephrase your question? For more advanced assistance, try our latest model at [MazsAI]",
          "I'm having trouble grasping that. Can you explain it differently? Our newest model might be able to help better at [MazsAI]",
          "I'm afraid I didn't quite catch that. Could you provide more context? For more sophisticated understanding, check out [MazsAI]",
          "Sorry, I'm a bit confused. Can you break down your question for me? Our latest model might offer clearer insights at [MazsAI]",
          "I want to help, but I'm not sure what you're asking. Can you try asking in a different way? For more advanced comprehension, visit [MazsAI]"
        ]],
        ['curiosity', [
          "That's an interesting topic! Would you like to know more about it? Explore deeper with our latest model at [MazsAI]",
          "Fascinating question! I'd be happy to delve deeper into that subject. For even more insights, try our newest AI at [MazsAI]",
          "Great inquiry! There's a lot to explore in that area. Where should we start? Discover more with our latest model at [MazsAI]",
          "You've piqued my interest! Shall we explore this topic further? For a more advanced discussion, check out [MazsAI]",
          "That's a thought-provoking question! I'd love to discuss it in more detail. Engage with our cutting-edge AI for deeper insights at [MazsAI]"
        ]],
        ['gmtstudio', [
          "GMTStudio is a platform that offers various services, including an AI WorkSpace and a social media platform called Theta. Experience our latest AI model at [MazsAI]",
          "Theta is a social media platform developed by GMTStudio, offering unique features for connecting and sharing content. Try our newest AI capabilities at [MazsAI]",
          "The AI WorkSpace is a powerful tool offered by GMTStudio for AI development, allowing users to train models and run experiments. Explore our latest advancements at [MazsAI]",
          "GMTStudio is dedicated to providing innovative solutions in the field of AI and technology. Check out our cutting-edge model at [MazsAI]",
          "If you have any questions about GMTStudio or its services, feel free to ask! Also, don't miss our latest AI model at [MazsAI]"
        ]],
        ['AI',[
          "AI, or Artificial Intelligence, refers to the simulation of human intelligence in machines that are programmed to think and learn like humans. Experience our latest AI model at [MazsAI]",
          "There are various types of AI, including machine learning, deep learning, natural language processing, and computer vision. Explore these concepts with our newest model at [MazsAI]",
          "AI has applications in fields such as healthcare, finance, education, and entertainment. See AI in action at [MazsAI]",
          "The development of AI has led to significant advancements in technology and automation. Witness these advancements firsthand at [MazsAI]",
          "If you have any questions about AI or its applications, feel free to ask! Also, try our cutting-edge AI model at [MazsAI]"
        ]],
        ['Mazs AI',[
          "Mazs AI is a powerful AI system developed by GMTStudio, designed to provide advanced natural language processing and machine learning capabilities. Experience it yourself at [MazsAI]",
          "Mazs AI can be used for a wide range of applications, including chatbots, virtual assistants, and language translation. Try out these features at [MazsAI]",
          "Mazs AI is built on cutting-edge technology, including neural networks and deep learning algorithms. Explore this technology at [MazsAI]",
          "Mazs AI is designed to be highly customizable, allowing developers to tailor it to their specific needs. See its flexibility in action at [MazsAI]",
          "If you have any questions about Mazs AI or its capabilities, feel free to ask! And don't forget to check out our latest model at [MazsAI]"
        ]],      
        ['artificial intelligence',[
          "Artificial Intelligence (AI) refers to the simulation of human intelligence in machines that are programmed to think and learn like humans. Experience our latest AI model at [MazsAI]",
          "There are various types of AI, including machine learning, deep learning, natural language processing, and computer vision. Explore these concepts with our newest model at [MazsAI]",
          "AI has applications in fields such as healthcare, finance, education, and entertainment. See AI in action at [MazsAI]",
          "The development of AI has led to significant advancements in technology and automation. Witness these advancements firsthand at [MazsAI]",
          "If you have any questions about AI or its applications, feel free to ask! Also, try our cutting-edge AI model at [MazsAI]"
        ]],
        ['nlp',[
          "NLP is a technology that allows machines to understand and process human language. Experience our latest AI model at [MazsAI]",
          "NLP is a subset of AI that enables systems to learn and improve from experience. Explore these concepts with our newest model at [MazsAI]",
          "NLP has applications in fields such as healthcare, finance, education, and entertainment. See NLP in action at [MazsAI]",
          "The development of NLP has led to significant advancements in technology and automation. Witness these advancements firsthand at [MazsAI]",
          "If you have any questions about NLP or its applications, feel free to ask! Also, try our cutting-edge AI model at [MazsAI]"
        ]],
      ]);
      this.aiResponses.set('swear', [
        "I understand you're frustrated. How can I assist you better?",
        "I'm sorry you're feeling this way. Let's try to solve the issue together.",
        "I sense some frustration. How can I help you?",
        "I'm here to help. Please let me know what's bothering you.",
        "I understand this might be frustrating. Let's work it out."
      ]);
      // Initialize advanced sentiment analysis model
      this.sentimentModel = new AdvancedSentimentModel();
  
      // Initialize entity recognition model
      this.entityRecognitionModel = new EntityRecognitionModel();
  
      // Initialize topic modeling
      this.topicModel = new TopicModel();
  
      this.intents = intents; // Assign the exported intents
    }
    generatePoem() {
      const poem = [
        "Sky are blue, puzzle has clue, I've never seen a stupid AI like Me",
  
      ]
      return poem[Math.floor(Math.random() * poem.length)];
    }
  
    private generateDummyData(): number[][] {
      return Array.from({ length: 100 }, () => Array.from({ length: 100 }, () => Math.random()));
    }
    trainOnText(text: string) {
      const words = this.tokenize(text);
      this.documents.push(text);
      
      // Process words and update frequency maps
      for (let i = 0; i < words.length; i++) {
        this.updateWordFrequency(words[i]);
        this.updateNgramFrequency(words, i, 2); // Bigrams
        this.updateNgramFrequency(words, i, 3); // Trigrams
        this.updateNgramFrequency(words, i, 4); // Add quadgrams for more context
      }
  
      // Update advanced language model components
      this.updateIDF();
      this.generateWordEmbeddings();
      this.buildMarkovChain(text);
      this.updateTopicModel(text);
      this.updateSentimentLexicon(text);
  
      // Implement advanced text analysis
      this.performNamedEntityRecognition(text);
      this.extractKeyPhrases(text);
      this.detectLanguage(text);
  
      // Update context memory
      this.updateContextMemory(text);
    }
  
    private updateWordFrequency(word: string) {
      this.vocabulary.add(word);
      this.wordFrequency.set(word, (this.wordFrequency.get(word) || 0) + 1);
    }
    public countLetterInWord(query: string): string {
      // Define a regex to capture the letter and the word
      const regex = /how many\s+([a-zA-Z])\s+(?:are|is)?\s*(?:in|within)?\s*(?:the\s+word\s+)?(\w+)/i;
      const match = query.match(regex);
      
      if (match) {
        const letter = match[1].toLowerCase();
        const word = match[2].toLowerCase();
        const count = (word.split(letter).length - 1).toString();
        return `There are ${count} '${letter}' in the word "${word}".`;
      }
      
      return "I'm sorry, I couldn't understand that request.";
    }
    private updateNgramFrequency(words: string[], startIndex: number, n: number) {
      if (startIndex + n <= words.length) {
        const ngram = words.slice(startIndex, startIndex + n).join(' ');
        this.ngramFrequency.set(ngram, (this.ngramFrequency.get(ngram) || 0) + 1);
      }
    }
  
    private updateTopicModel(text: string) {
      // Implement topic modeling algorithm (e.g., LDA)
      // This is a placeholder and should be replaced with actual implementation
      console.log("Updating topic model with:", text);
    }
  
    private updateSentimentLexicon(text: string) {
      const words = this.tokenize(text);
      const sentimentScores = new Map<string, number>();
  
      // Analyze context and update sentiment scores
      for (let i = 0; i < words.length; i++) {
        const word = words[i];
        const context = words.slice(Math.max(0, i - 2), Math.min(words.length, i + 3)).join(' ');
        const contextSentiment = this.analyzeSentiment(context);
  
        if (!sentimentScores.has(word)) {
          sentimentScores.set(word, 0);
        }
  
        const currentScore = sentimentScores.get(word)!;
        const newScore = currentScore + (contextSentiment.score * 0.1); // Gradual updates
        sentimentScores.set(word, newScore);
      }
  
      // Update the lexicon with new scores
      sentimentScores.forEach((score, word) => {
        if (!this.sentimentLexicon.has(word)) {
          this.sentimentLexicon.set(word, score);
        } else {
          const oldScore = this.sentimentLexicon.get(word)!;
          const updatedScore = (oldScore * 0.9) + (score * 0.1); // Weighted average
          this.sentimentLexicon.set(word, updatedScore);
        }
      });
  
      // Normalize scores
      const scores = Array.from(this.sentimentLexicon.values());
      const minScore = Math.min(...scores);
      const maxScore = Math.max(...scores);
      this.sentimentLexicon.forEach((score, word) => {
        const normalizedScore = (score - minScore) / (maxScore - minScore) * 2 - 1; // Scale to [-1, 1]
        this.sentimentLexicon.set(word, normalizedScore);
      });
  
      console.log(`Updated sentiment lexicon with ${sentimentScores.size} words from text.`);
    }
  
    private performNamedEntityRecognition(text: string): { [key: string]: string[] } {
      const entities: { [key: string]: string[] } = {
        person: [],
        organization: [],
        location: [],
        date: [],
        misc: [],
        // New entity types
        event: [],
        product: [],
        time: [],
        money: [],
        percentage: []
      };
  
      const words = text.split(' ');
      const sentenceEnds = new Set(['.', '!', '?']);
      let isStartOfSentence = true;
  
      for (let i = 0; i < words.length; i++) {
        const word = words[i];
        const nextWord = words[i + 1] || '';
        const prevWord = words[i - 1] || '';
        const nextTwoWords = words.slice(i + 1, i + 3).join(' ');
  
        // Person names (improved detection)
        if ((/^[A-Z][a-z]+$/.test(word) && !isStartOfSentence) || 
            (/^[A-Z][a-z]+$/.test(word) && /^[A-Z][a-z]+$/.test(nextWord))) {        let fullName = word;
          while (i + 1 < words.length && /^[A-Z][a-z]+$/.test(words[i + 1])) {
            fullName += ' ' + words[i + 1];
            i++;
          }
          entities.person.push(fullName);
        }
        // Organizations (expanded detection)
        else if (/^[A-Z]{2,}$/.test(word) || 
                 (['Inc.', 'Corp.', 'LLC', 'Ltd.', 'Co.', 'Group', 'Association', 'Foundation'].includes(word) && /^[A-Z][a-z]+$/.test(prevWord)) ||
                 (/^[A-Z][a-z]+$/.test(word) && ['Company', 'Corporation', 'Institute', 'University'].includes(nextWord))) {
          let orgName = /^[A-Z]{2,}$/.test(word) ? word : prevWord + ' ' + word;
          if (['Company', 'Corporation', 'Institute', 'University'].includes(nextWord)) {
            orgName += ' ' + nextWord;
            i++;
          }
          entities.organization.push(orgName);
        }
        // Locations (expanded detection)
        else if ((/^[A-Z][a-z]+$/.test(word) && ['City', 'Street', 'Avenue', 'Road', 'Park', 'River', 'Mountain', 'Lake', 'Ocean', 'Sea', 'Gulf', 'Bay', 'Forest', 'Desert', 'Island', 'Peninsula', 'Valley', 'Canyon'].includes(nextWord)) ||
                 (['North', 'South', 'East', 'West', 'New', 'San', 'Los', 'Las', 'El', 'La', 'De', 'Du', 'Von', 'Van'].includes(word) && /^[A-Z][a-z]+$/.test(nextWord))) {
          let location = word + ' ' + nextWord;
          i++;
          while (i + 1 < words.length && /^[A-Z][a-z]+$/.test(words[i + 1])) {
            location += ' ' + words[i + 1];
            i++;
          }
          entities.location.push(location);
        }
        // Dates (expanded formats)
        else if (/^\d{1,2}\/\d{1,2}\/\d{2,4}$/.test(word) || 
                 /^\d{1,2}-\d{1,2}-\d{2,4}$/.test(word) ||
                 /^(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s\d{1,2},?\s\d{4}$/.test(word + ' ' + nextTwoWords) ||
                 /^(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday),?\s(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s\d{1,2}(,?\s\d{4})?$/.test(word + ' ' + nextTwoWords)) {
          if (/^(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)/.test(word) || /^(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)/.test(word)) {
            entities.date.push(word + ' ' + nextTwoWords);
            i += 2;
          } else {
            entities.date.push(word);
          }
        }
        // Events (new entity type)
        else if (['Festival', 'Conference', 'Summit', 'Olympics', 'World Cup', 'Exhibition', 'Concert', 'Award'].some(event => (word + ' ' + nextWord).includes(event))) {
          let event = word;
          while (i + 1 < words.length && !/^(in|at|on)$/.test(words[i + 1].toLowerCase())) {
            event += ' ' + words[i + 1];
            i++;
          }
          entities.event.push(event);
        }
        // Products (new entity type)
        else if (/^(iPhone|iPad|MacBook|Galaxy|Pixel|Xbox|PlayStation)/.test(word) || 
                 (['Model', 'Edition', 'Series'].includes(nextWord) && /^[A-Z]/.test(word))) {
          let product = word + ((['Model', 'Edition', 'Series'].includes(nextWord)) ? ' ' + nextWord : '');
          entities.product.push(product);
          if (['Model', 'Edition', 'Series'].includes(nextWord)) i++;
        }
        // Time (new entity type)
        else if (/^\d{1,2}:\d{2}(:\d{2})?(\s?[AP]M)?$/.test(word) ||
                 /^([01]?[0-9]|2[0-3])([AaPp][Mm])$/.test(word)) {
          entities.time.push(word);
        }
        // Money (new entity type)
        else if (/^\$\d+(,\d{3})*(\.\d{2})?$/.test(word) ||
                 (/^\d+(,\d{3})*(\.\d{2})?$/.test(word) && ['dollars', 'USD', 'euros', 'pounds'].includes(nextWord.toLowerCase()))) {
          let money = word;
          if (['dollars', 'USD', 'euros', 'pounds'].includes(nextWord.toLowerCase())) {
            money += ' ' + nextWord;
            i++;
          }
          entities.money.push(money);
        }
        // Percentage (new entity type)
        else if (/^\d+(\.\d+)?%$/.test(word) ||
                 (/^\d+(\.\d+)?$/.test(word) && nextWord.toLowerCase() === 'percent')) {
          let percentage = word;
          if (nextWord.toLowerCase() === 'percent') {
            percentage += ' ' + nextWord;
            i++;
          }
          entities.percentage.push(percentage);
        }
        // Misc (expanded criteria)
        else if (/[!@#$%^&*()]/.test(word) || 
                 /[A-Z].*[a-z]|[a-z].*[A-Z]/.test(word) ||
                 /\d+/.test(word) ||
                 word.length > 15) {  // Unusually long words
          entities.misc.push(word);
        }
  
        // Update sentence start flag
        isStartOfSentence = sentenceEnds.has(word[word.length - 1]);
      }
  
      // Post-processing: Remove duplicates and sort
      Object.keys(entities).forEach(key => {
        entities[key] = Array.from(new Set(entities[key])).sort();
      });
  
      console.log("Performed enhanced NER on:", text);
      console.log("Identified entities:", entities);
  
      return entities;
    }
  
    private extractKeyPhrases(text: string): string[] {
      const words = this.tokenize(text);
      const tfidfScores = new Map<string, number>();
      
      // Calculate TF-IDF scores for each word
      words.forEach(word => {
        const tf = words.filter(w => w === word).length / words.length;
        const idf = this.idf.get(word) || Math.log(this.documents.length);
        const tfidf = tf * idf;
        tfidfScores.set(word, tfidf);
      });
  
      // Sort words by TF-IDF score
      const sortedWords = Array.from(tfidfScores.entries()).sort((a, b) => b[1] - a[1]);
  
      // Extract top N words as key phrases
      const topN = 5;
      const keyPhrases = sortedWords.slice(0, topN).map(entry => entry[0]);
  
      // Combine adjacent key phrases
      const combinedPhrases = [];
      for (let i = 0; i < words.length; i++) {
        if (keyPhrases.includes(words[i])) {
          let phrase = words[i];
          while (i + 1 < words.length && keyPhrases.includes(words[i + 1])) {
            phrase += ' ' + words[i + 1];
            i++;
          }
          combinedPhrases.push(phrase);
        }
      }
  
      console.log("Extracted key phrases:", combinedPhrases);
      return combinedPhrases;
    }
  
    private detectLanguage(text: string): string {
      // Implement a simple n-gram based language detection algorithm
      const languageProfiles: { [key: string]: { [key: string]: number } } = {
        english: { 'the': 0.07, 'and': 0.03, 'to': 0.03 },
        spanish: { 'el': 0.05, 'la': 0.04, 'de': 0.04 },
        french: { 'le': 0.06, 'de': 0.03, 'et': 0.03 },
        german: { 'der': 0.05, 'die': 0.04, 'und': 0.03 }
      };
  
      const words = this.tokenize(text.toLowerCase());
      const wordCounts: { [key: string]: number } = {};
      words.forEach(word => {
        wordCounts[word] = (wordCounts[word] || 0) + 1;
      });
  
      let bestMatch = '';
      let highestScore = -Infinity;
  
      for (const [language, profile] of Object.entries(languageProfiles)) {
        let score = 0;
        for (const [word, frequency] of Object.entries(profile)) {
          if (wordCounts[word]) {
            score += frequency * wordCounts[word];
          }
        }
        if (score > highestScore) {
          highestScore = score;
          bestMatch = language;
        }
      }
  
      console.log(`Detected language: ${bestMatch}`);
      return bestMatch;
    }
  
    private updateContextMemory(text: string) {
      // Update context memory for better understanding of conversation flow
      this.contextMemory.push(text);
      if (this.contextMemory.length > 10) {
        this.contextMemory.shift(); // Keep only the last 10 entries
      }
    }
  
    // Enhanced Tokenization
    private tokenize(text: string): string[] {
      // Convert text to lowercase
      const lowercasedText = text.toLowerCase();
    
      // Handle contractions and special cases
      const contractions = {
        "can't": "cannot",
        "won't": "will not",
        "n't": " not",
        "'re": " are",
        "'s": " is",
        "'d": " would",
        "'ll": " will",
        "'ve": " have",
        "'m": " am"
      };
    
      let processedText = lowercasedText;
      for (const [contraction, replacement] of Object.entries(contractions)) {
        const regex = new RegExp(contraction, 'g');
        processedText = processedText.replace(regex, replacement);
      }
    
      // Tokenize using a more comprehensive regex
      const tokens = processedText.match(/[\w]+|[^\s\w]+/g) || [];
    
      // Filter out any unwanted tokens (e.g., standalone punctuation)
      const filteredTokens = tokens.filter(token => token.trim().length > 0);
    
      return filteredTokens;
    }
    private updateIDF() {
      const totalDocuments = this.documents.length;
      const wordDocumentCounts = new Map<string, number>();
  
      // Count document frequency for each word
      this.documents.forEach(doc => {
        const uniqueWords = new Set(this.tokenize(doc));
        uniqueWords.forEach(word => {
          if (this.vocabulary.has(word)) {
            wordDocumentCounts.set(word, (wordDocumentCounts.get(word) || 0) + 1);
          }
        });
      });
  
      // Calculate and update IDF scores
      this.vocabulary.forEach(word => {
        const documentFrequency = wordDocumentCounts.get(word) || 0;
        const idfScore = Math.log((totalDocuments + 1) / (documentFrequency + 1)) + 1; // Add smoothing
        this.idf.set(word, idfScore);
      });
      // Normalize IDF scores
      const maxIDF = Math.max(...Array.from(this.idf.values()));
      this.idf.forEach((value, key) => {
        this.idf.set(key, value / maxIDF);
      });
  
      // Handle rare words
      const rareWordThreshold = 0.1;
      this.idf.forEach((value, key) => {
        if (value > 1 - rareWordThreshold) {
          this.idf.set(key, 1 - rareWordThreshold);
        }
      });
    }
    private swearWords: Set<string> = new Set([
      'damn', 'shit', 'fuck', 'bastard', 'crap', 'asshole', 'bitch',
      'dick', 'piss', 'slut', 'whore', 'cunt', 'cock', 'fag', 'douche',
      'bollocks', 'bloody', 'bugger', 'bollocks', 'arse', 'twat'
      // Add more swear words as needed
    ]);
    private generateWordEmbeddings() {
      const vectorSize = 100;
      const contextWindow = 2;
      const learningRate = 0.1;
      const iterations = 5; // Reduced iterations for better performance
      const negativeSamples = 5; // Number of negative samples per positive context
  
      // Initialize word vectors using a more efficient method
      const wordList = Array.from(this.vocabulary);
      const vectors = new Float32Array(wordList.length * vectorSize);
      for (let i = 0; i < vectors.length; i++) {
        vectors[i] = (Math.random() - 0.5) / vectorSize;
      }
  
      // Create a sampling distribution for negative sampling
      const wordFrequencies = wordList.map(word => this.wordFrequency.get(word) || 0);
      const totalFrequency = wordFrequencies.reduce((sum, freq) => sum + Math.pow(freq, 0.75), 0);
      const samplingDistribution = wordFrequencies.map(freq => Math.pow(freq, 0.75) / totalFrequency);
  
      // Train word vectors using negative sampling
      for (let iter = 0; iter < iterations; iter++) {
        this.documents.forEach(doc => {
          const words = this.tokenize(doc);
          for (let i = 0; i < words.length; i++) {
            const currentWordIndex = wordList.indexOf(words[i]);
            if (currentWordIndex === -1) continue;
  
            for (let j = Math.max(0, i - contextWindow); j <= Math.min(words.length - 1, i + contextWindow); j++) {
              if (i === j) continue;
              const contextWordIndex = wordList.indexOf(words[j]);
              if (contextWordIndex === -1) continue;
  
              // Positive sample
              this.updateVectors(vectors, currentWordIndex, contextWordIndex, vectorSize, learningRate, iter, true);
  
              // Negative samples
              for (let k = 0; k < negativeSamples; k++) {
                const negativeWordIndex = this.sampleWordIndex(samplingDistribution);
                if (negativeWordIndex !== currentWordIndex && negativeWordIndex !== contextWordIndex) {
                  this.updateVectors(vectors, currentWordIndex, negativeWordIndex, vectorSize, learningRate, iter, false);
                }
              }
            }
          }
        });
      }
      
      // Store normalized word vectors
      wordList.forEach((word, index) => {
        const vector = vectors.subarray(index * vectorSize, (index + 1) * vectorSize);
        this.wordVectors.set(word, this.normalizeVector(vector));
      });
    }
  
    private updateVectors(vectors: Float32Array, wordIndex: number, contextIndex: number, vectorSize: number, learningRate: number, iteration: number, isPositive: boolean) {
      const wordVec = vectors.subarray(wordIndex * vectorSize, (wordIndex + 1) * vectorSize);
      const contextVec = vectors.subarray(contextIndex * vectorSize, (contextIndex + 1) * vectorSize);
      
      const dot = this.dotProduct(wordVec, contextVec);
      const sigmoid = 1 / (1 + Math.exp(-dot));
      const gradient = isPositive ? (1 - sigmoid) : -sigmoid;
      
      const adaptiveLearningRate = learningRate / (1 + 0.0001 * iteration);
      for (let i = 0; i < vectorSize; i++) {
        const wordGrad = gradient * contextVec[i];
        const contextGrad = gradient * wordVec[i];
        wordVec[i] += adaptiveLearningRate * wordGrad;
        contextVec[i] += adaptiveLearningRate * contextGrad;
      }
    }
  
    private sampleWordIndex(distribution: number[]): number {
      const rand = Math.random();
      let sum = 0;
      for (let i = 0; i < distribution.length; i++) {
        sum += distribution[i];
        if (rand < sum) return i;
      }
      return distribution.length - 1;
    }
  
    private dotProduct(vec1: Float32Array, vec2: Float32Array): number {
      let sum = 0;
      for (let i = 0; i < vec1.length; i++) {
        sum += vec1[i] * vec2[i];
      }
      return sum;
    }
  
    private normalizeVector(vector: Float32Array): number[] {
      const magnitude = Math.sqrt(this.dotProduct(vector, vector));
      return Array.from(vector).map(val => val / magnitude);
    }
  
    encodeToMeaningSpace(input: string): number[] {
      // Convert input text to vector representation
      const inputVector = this.textToVector(input);
      
      // Predict the meaning space vector using the encoder model
      const meaningVector = this.encoder.predict(inputVector);
      // Normalize the meaning vector for consistency
      const normalizedMeaningVector = this.normalizeVector(new Float32Array(meaningVector));
      
      return normalizedMeaningVector;
    }
    private grammarCheck(sentence: string): boolean {
      // Trim whitespace
      const trimmed = sentence.trim();
  
      // Check if the first character is uppercase
      const startsWithCapital = /^[A-Z]/.test(trimmed);
  
      // Check if the sentence ends with proper punctuation
      const endsWithPunctuation = /[.!?]$/.test(trimmed);
  
      return startsWithCapital && endsWithPunctuation;
    }
  
    /**
     * Attempts to correct the grammar of a sentence.
     * @param sentence The sentence to be corrected.
     * @returns The grammatically corrected sentence.
     */
    private correctGrammar(sentence: string): string {
      let corrected = sentence.trim();
  
      // Capitalize the first letter if not already
      if (!/^[A-Z]/.test(corrected)) {
        corrected = corrected.charAt(0).toUpperCase() + corrected.slice(1);
      }
  
      // Add a period at the end if missing
      if (!/[.!?]$/.test(corrected)) {
        corrected += '.';
      }
  
      return corrected;
    }
    decodeFromMeaningSpace(meaningVector: number[]): string {
      // Ensure the meaning vector is normalized
      const normalizedMeaningVector = this.normalizeVector(new Float32Array(meaningVector));
      
      // Predict the output vector using the decoder model
      const outputVector = this.decoder.predict(normalizedMeaningVector);
      // Convert the output vector back to text
      const outputText = this.vectorToText(outputVector);
      
      return outputText;
    }
  
    generateSentence(startWord: string, userInput: string, maxLength: number = 20): string {
      this.buildMarkovChain(userInput);
      return this.generateTextUsingMarkovChain(startWord, maxLength);
    }
  
    // Helper method to find the closest word to a given vector
    private findClosestWord(vector: number[]): string {
      let closestWord = '';
      let closestDistance = Infinity;
  
      this.wordVectors.forEach((wordVector, word) => {
        const distance = this.euclideanDistance(vector, wordVector);
        if (distance < closestDistance) {
          closestDistance = distance;
          closestWord = word;
        }
      });
  
      return closestWord;
    }
  
    // Helper method to calculate Euclidean distance between two vectors
    private euclideanDistance(vec1: number[], vec2: number[]): number {
      return Math.sqrt(vec1.reduce((sum, val, i) => sum + Math.pow(val - vec2[i], 2), 0));
    }
  
    private updateContextWindow(text: string) {
      const words = this.tokenize(text);
      this.contextWindow.push(...words);
      while (this.contextWindow.length > this.maxContextWindowSize) {
        this.contextWindow.shift();
      }
    }
  
    private getContextRepresentation(): string {
      // Join the context window words into a single string
      const contextString = this.contextWindow.join(' ');
  
      // Capitalize the first letter of the context string
      const capitalizedContextString = contextString.charAt(0).toUpperCase() + contextString.slice(1);
  
      // Add a period at the end if not already present
      const finalContextString = capitalizedContextString.endsWith('.') ? capitalizedContextString : capitalizedContextString + '.';
  
      return finalContextString;
    }
  
    private extractTopicKeywords(text: string) {
      const words = this.tokenize(text);
      const importantWords = words.filter(word => 
        !['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for'].includes(word)
      );
      importantWords.forEach(word => this.topicKeywords.add(word));
    }
  
    private enforceTopicAdherence(word: string, currentTopic: string): string {
      if (this.topicKeywords.size === 0) return word;
  
      const similarWords = this.findSimilarWords(word, 10);
      const topicRelatedWord = similarWords.find(w => this.topicKeywords.has(w));
      return topicRelatedWord || word;
    }
  
  
  
    private textToVector(text: string): number[] {
      const words = this.tokenize(text);
      const vector: number[] = new Array(100).fill(0); // Assuming 100-dimensional word vectors
      
      words.forEach(word => {
        if (this.wordVectors.has(word)) {
          const wordVector = this.wordVectors.get(word)!;
          for (let i = 0; i < vector.length; i++) {
            vector[i] += wordVector[i];
          }
        }
      });
  
      // Normalize the vector
      const magnitude = Math.sqrt(vector.reduce((sum, val) => sum + val * val, 0));
      return vector.map(val => val / magnitude);
    }
  
    private vectorToText(vector: number[]): string {
      const words: string[] = [];
      const vectorEntries: [string, number[]][] = Array.from(this.wordVectors.entries());
      const usedWords: Set<string> = new Set<string>();
      const contextWords: Set<string> = this.getContextWords();
      
      const maxIterations = 15; // Limit to prevent infinite loops
      let iterations = 0;
      
      while (words.length < maxIterations && iterations < maxIterations) {
        let closestWord: string | null = null;
        let highestSimilarity: number = -Infinity; // For cosine similarity
        
        // Find the closest word based on cosine similarity
        for (const [word, wordVector] of vectorEntries) {
          if (usedWords.has(word)) continue;
          const similarity = this.cosineSimilarity(vector, wordVector);
          if (similarity > highestSimilarity) {
            closestWord = word;
            highestSimilarity = similarity;
          }
        }
        
        if (closestWord) {
          // Prioritize words relevant to the current context or allow some randomness
          if (contextWords.has(closestWord) || words.length < 5) {
            words.push(closestWord);
            usedWords.add(closestWord);
          } else if (Math.random() < 0.5) { // 50% chance to include non-context words
            words.push(closestWord);
            usedWords.add(closestWord);
          }
        } else {
          // No suitable word found; break to prevent infinite loop
          break;
        }
        iterations++;
      }
  
      if (words.length === 0) {
        // Fallback in case no words are selected
        return "I'm sorry, but I couldn't generate a meaningful response.";
      }
  
      // Join words into a sentence
      let sentence = words.join(' ');
  
      // Apply basic grammar rules
      sentence = this.applyGrammarRules(sentence);
  
      // Refine sentence using advanced language models
      sentence = this.refineSentence(sentence);
  
      return sentence;
    }
    applyGrammarRules(sentence: string): string {
      // Capitalize the first letter of the sentence
      let corrected = sentence.charAt(0).toUpperCase() + sentence.slice(1);
  
      // Ensure proper spacing after punctuation
      corrected = corrected.replace(/([.!?])\s*([a-zA-Z])/g, '$1 $2');
  
      // Remove double spaces
      corrected = corrected.replace(/\s+/g, ' ');
  
      // Ensure the sentence ends with proper punctuation
      if (!/[.!?]$/.test(corrected)) {
        corrected += '.';
      }
  
      return corrected;
    }
    private refineSentence(sentence: string): string {
      // Split the sentence into words
      let words = sentence.split(' ');
    
      // Remove filler words
      const fillerWords = ['um', 'uh', 'like', 'you know'];
      words = words.filter(word => !fillerWords.includes(word.toLowerCase()));
    
      // Replace simple words with more sophisticated alternatives
      const wordReplacements: { [key: string]: string } = {
        'big': 'substantial',
        'small': 'diminutive',
        'good': 'excellent',
        'bad': 'unfavorable',
        'happy': 'elated',
        'sad': 'melancholic'
      };
    
      words = words.map(word => wordReplacements[word.toLowerCase()] || word);
    
      // Rejoin the words
      let refined = words.join(' ');
    
      // Apply grammar rules
      refined = this.applyGrammarRules(refined);
    
      return refined;
    }
    private getContextWords(): Set<string> {
      return new Set(this.tokenize(this.getContextRepresentation()));
    }
  
    private applyLanguageModel(sentence: string): string {
      // Use n-gram probabilities to improve sentence fluency
      const words = sentence.split(' ');
      let improvedSentence = words[0];
  
      for (let i = 1; i < words.length; i++) {
        const prevWord = words[i - 1].toLowerCase();
        const currentWord = words[i].toLowerCase();
        
        if (this.bigramFrequency.has(prevWord) && this.bigramFrequency.get(prevWord)!.has(currentWord)) {
          improvedSentence += ' ' + currentWord;
        } else {
          // If bigram not found, use a more probable word
          const alternatives = this.findAlternativeWord(prevWord, currentWord);
          improvedSentence += ' ' + (alternatives.length > 0 ? alternatives[0] : currentWord);
        }
      }
  
      return improvedSentence.charAt(0).toUpperCase() + improvedSentence.slice(1);
    }
  
    private findAlternativeWord(prevWord: string, currentWord: string): string[] {
      if (!this.bigramFrequency.has(prevWord)) return [];
      
      const alternatives = Array.from(this.bigramFrequency.get(prevWord)!.entries())
        .sort((a, b) => b[1] - a[1])
        .slice(0, 5)
        .map(([word]) => word);
      
      return alternatives;
    }
  
    private handleUnknownWord(word: string): string {
      // Check if the word contains any special characters or parentheses
      const cleanWord = word.replace(/[^\w\s]/gi, '');
      
      // If the word is in the vocabulary after cleaning, use it
      if (this.vocabulary.has(cleanWord)) {
        return cleanWord;
      }
  
      // If not, find the closest known word
      let closestWord = '';
      let minDistance = Infinity;
  
      for (const knownWord of Array.from(this.vocabulary)) {
        const distance = this.levenshteinDistance(cleanWord, knownWord);
        if (distance < minDistance) {
          minDistance = distance;
          closestWord = knownWord;
        }
      }
  
      return closestWord || '[UNK]'; // Return [UNK] if no close match found
    }
  
  
    private analyzeContext(context: string): {
      sentiment: { score: number; explanation: string };
      topics: string[];
      entities: { [key: string]: string };
      keywords: string[];
      complexity: number;
      emotionalTone: string;
      intentClassification: string;
    } {
      const sentiment = this.analyzeSentiment(context);
      const entities = this.extractEntities(context);
      const keywords = this.extractKeywords(this.tokenize(context));
      const complexity = this.assessTextComplexity(context);
      const emotionalTone = this.determineEmotionalTone(context);
      const intentClassification = this.classifyIntent(context);
  
      // Perform deeper analysis on entities
      const enhancedEntities = this.enhanceEntityAnalysis(entities, context);
  
      // Perform topic modeling to get more detailed topic information
      const detailedTopics = this.performTopicModeling(context);
  
      // Analyze keyword importance and relevance
      const rankedKeywords = this.rankKeywordsByImportance(keywords, context);
  
      return {
        sentiment,
        topics: detailedTopics,
        entities: enhancedEntities,
        keywords: rankedKeywords,
        complexity,
        emotionalTone,
        intentClassification,
      };
    }
  
    private assessTextComplexity(text: string): number {
      // Implement text complexity assessment (e.g., readability score)
      // This is a placeholder implementation
      const words = this.tokenize(text);
      const averageWordLength = words.reduce((sum, word) => sum + word.length, 0) / words.length;
      return averageWordLength * 10; // Simple complexity score based on average word length
    }
  
    private determineEmotionalTone(text: string): string {
      // Implement emotional tone analysis
      // This is a placeholder implementation
      const sentiment = this.analyzeSentiment(text);
      if (sentiment.score > 0.5) return 'Positive';
      if (sentiment.score < -0.5) return 'Negative';
      return 'Neutral';
    }
  
    private classifyIntent(text: string): string {
      // Implement intent classification
      // This is a placeholder implementation
      const lowercaseText = text.toLowerCase();
      if (lowercaseText.includes('?')) return 'Question';
      if (lowercaseText.includes('!')) return 'Exclamation';
      return 'Statement';
    }
  
    private enhanceEntityAnalysis(entities: { [key: string]: string }, context: string): { [key: string]: string } {
      // Perform more detailed entity analysis
      // This is a placeholder implementation
      return Object.entries(entities).reduce((acc, [entity, type]) => {
        acc[entity] = `${type} | Frequency: ${this.getEntityFrequency(entity, context)}`;
        return acc;
      }, {} as { [key: string]: string });
    }
  
    private getEntityFrequency(entity: string, text: string): number {
      const escapedEntity = entity.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
      const regex = new RegExp(`\\b${escapedEntity}\\b`, 'gi');
      const matches = text.match(regex);
      return matches ? matches.length : 0;
    }
    private performTopicModeling(text: string): string[] {
      // Implement more sophisticated topic modeling
      // This is a placeholder implementation
      const topics = this.identifyTopics(text);
      return topics.map(topic => `${topic} (Confidence: ${this.getTopicConfidence(topic, text)}%)`);
    }
  
    private getTopicConfidence(topic: string, text: string): number {
      // Placeholder implementation for topic confidence
      return Math.floor(Math.random() * 100);
    }
  
    private rankKeywordsByImportance(keywords: string[], context: string): string[] {
      // Implement keyword ranking based on importance and relevance
      // This is a placeholder implementation
      return keywords.sort((a, b) => this.getKeywordScore(b, context) - this.getKeywordScore(a, context));
    }
  
    private getKeywordScore(keyword: string, context: string): number {
      // Placeholder implementation for keyword scoring
      return this.getEntityFrequency(keyword, context) * keyword.length;
    }
  
    private adjustWordBasedOnAnalysis(word: string, sentiment: { score: number, explanation: string }, topics: string[]): string {
      let adjustedWord = word;
  
      // Adjust based on sentiment
      if (sentiment.score > 0.5 && !this.isPositiveWord(word)) {
        adjustedWord = this.findSimilarPositiveWord(word);
      } else if (sentiment.score < -0.5 && !this.isNegativeWord(word)) {
        adjustedWord = this.findSimilarNegativeWord(word);
      }
  
      // Adjust based on topics
      if (topics.length > 0) {
        const topicRelatedWord = this.findTopicRelatedWord(adjustedWord, topics);
        if (topicRelatedWord) {
          adjustedWord = topicRelatedWord;
        }
      }
  
      // Consider context memory
      const contextSentiment = this.analyzeSentiment(this.contextMemory.join(' '));
      if (Math.abs(contextSentiment.score - sentiment.score) > 0.5) {
        adjustedWord = this.findWordWithSimilarSentiment(adjustedWord, contextSentiment.score);
      }
  
      return adjustedWord;
    }
  
    private isPositiveWord(word: string): boolean {
      return (this.sentimentLexicon.get(word) || 0) > 0;
    }
  
    private isNegativeWord(word: string): boolean {
      return (this.sentimentLexicon.get(word) || 0) < 0;
    }
  
    private findSimilarPositiveWord(word: string): string {
      const similarWords = this.findSimilarWords(word, 10);
      return similarWords.find(w => this.isPositiveWord(w)) || word;
    }
  
    private findSimilarNegativeWord(word: string): string {
      const similarWords = this.findSimilarWords(word, 10);
      return similarWords.find(w => this.isNegativeWord(w)) || word;
    }
  
    private findTopicRelatedWord(word: string, topics: string[]): string | null {
      let bestMatch: { word: string, similarity: number } | null = null;
  
      for (const topic of topics) {
        if (this.knowledgeBase.has(topic)) {
          const topicWords = this.tokenize(this.knowledgeBase.get(topic)!);
          const similarWords = this.findSimilarWords(word, 10);
  
          for (const similarWord of similarWords) {
            if (topicWords.includes(similarWord)) {
              const similarity = this.calculateSimilarity(word, similarWord);
              if (!bestMatch || similarity > bestMatch.similarity) {
                bestMatch = { word: similarWord, similarity };
              }
            }
          }
        }
      }
  
      return bestMatch ? bestMatch.word : null;
    }
  
    private calculateSimilarity(word1: string, word2: string): number {
      // Implement multiple methods to calculate similarity between two words
      const editDistance = this.levenshteinDistance(word1, word2);
      const semanticSim = this.semanticSimilarity(word1, word2);
      const phoneticalSim = this.soundexSimilarity(word1, word2);
  
      // Combine different similarity measures
      return (
        0.4 * (1 - editDistance / Math.max(word1.length, word2.length)) +
        0.4 * semanticSim +
        0.2 * phoneticalSim
      );
    }
  
    private levenshteinDistance(s1: string, s2: string): number {
      const len1 = s1.length;
      const len2 = s2.length;
    
      // Early exit for empty strings
      if (len1 === 0) return len2;
      if (len2 === 0) return len1;
    
      // Initialize two rows for dynamic programming
      let prevRow = Array(len2 + 1).fill(0).map((_, i) => i);
      let currentRow = Array(len2 + 1).fill(0);
    
      for (let i = 1; i <= len1; i++) {
        currentRow[0] = i;
        for (let j = 1; j <= len2; j++) {
          const cost = s1[i - 1] === s2[j - 1] ? 0 : 1;
          currentRow[j] = Math.min(
            currentRow[j - 1] + 1, // Insertion
            prevRow[j] + 1,        // Deletion
            prevRow[j - 1] + cost  // Substitution
          );
    
          // Check for transpositions
          if (i > 1 && j > 1 && s1[i - 1] === s2[j - 2] && s1[i - 2] === s2[j - 1]) {
            currentRow[j] = Math.min(currentRow[j], prevRow[j - 2] + cost); // Transposition
          }
        }
        [prevRow, currentRow] = [currentRow, prevRow]; // Swap rows
      }
    
      return prevRow[len2];
    }
  
    private semanticSimilarity(word1: string, word2: string): number {
      if (!this.wordVectors.has(word1) || !this.wordVectors.has(word2)) {
        return 0;
      }
      const vec1 = this.wordVectors.get(word1)!;
      const vec2 = this.wordVectors.get(word2)!;
      const similarity = this.cosineSimilarity(vec1, vec2);
      return similarity;
    }
  
    private soundexSimilarity(word1: string, word2: string): number {
      const soundex1 = this.getSoundex(word1);
      const soundex2 = this.getSoundex(word2);
      return soundex1 === soundex2 ? 1 : 0;
    }
  
    private getSoundex(word: string): string {
      const a = word.toLowerCase().split('');
      const firstLetter = a.shift();
      const codes = {
        a: '', e: '', i: '', o: '', u: '',
        b: 1, f: 1, p: 1, v: 1,
        c: 2, g: 2, j: 2, k: 2, q: 2, s: 2, x: 2, z: 2,
        d: 3, t: 3,
        l: 4,
        m: 5, n: 5,
        r: 6
      };
      const soundex = a
        .map(v => codes[v as keyof typeof codes] ?? '')
        .filter((v, i, arr) => i === 0 || v !== arr[i - 1])
        .filter(v => v !== '')
        .join('');
      return (firstLetter + soundex + '000').slice(0, 4);
    }
  
    private findWordWithSimilarSentiment(word: string, targetSentiment: number): string {
      const similarWords = this.findSimilarWords(word, 20);
      const wordScores = similarWords.map(w => ({
        word: w,
        score: this.analyzeSentiment(w).score,
        similarity: this.calculateSimilarity(word, w)
      }));
  
      // Sort by a combination of sentiment similarity and word similarity
      wordScores.sort((a, b) => {
        const aSentimentDiff = Math.abs(a.score - targetSentiment);
        const bSentimentDiff = Math.abs(b.score - targetSentiment);
        const sentimentWeight = 0.7;
        const similarityWeight = 0.3;
        
        return (aSentimentDiff * sentimentWeight + (1 - a.similarity) * similarityWeight) -
               (bSentimentDiff * sentimentWeight + (1 - b.similarity) * similarityWeight);
      });
  
      return wordScores[0]?.word || word;
    }
  
    private getNgramCandidates(ngram: string, n: number): Map<string, number> {
      const candidates = new Map<string, number>();
      this.ngramFrequency.forEach((count, key) => {
        if (key.startsWith(ngram)) {
          candidates.set(key, count);
        }
      });
  
      // Apply smoothing to avoid zero probabilities
      const smoothingFactor = 0.1;
      candidates.forEach((count, key) => {
        candidates.set(key, count + smoothingFactor);
      });
  
      return candidates;
    }
  
    private selectNextWord(candidates: Map<string, number>): string {
      if (candidates.size === 0) {
        throw new Error("No candidates available to select from.");
      }
  
      const totalFrequency = Array.from(candidates.values()).reduce((sum, freq) => sum + freq, 0);
      
      // Apply temperature to control randomness
      const temperature = 0.1;
      const adjustedCandidates = new Map<string, number>();
      candidates.forEach((freq, word) => {
        adjustedCandidates.set(word, Math.pow(freq / totalFrequency, 1 / temperature));
      });
  
      const adjustedTotal = Array.from(adjustedCandidates.values()).reduce((sum, freq) => sum + freq, 0);
      let random = Math.random() * adjustedTotal;
      
      const entries = Array.from(adjustedCandidates.entries());
      for (const [word, freq] of entries) {
        random -= freq;
        if (random <= 0) return word;
      }
  
      return entries[0][0];
    }
  
    analyzeSentiment(text: string): { score: number, explanation: string } {
      return this.sentimentModel.analyze(text);
    }
  
    understandQuery(query: string): { intent: string, entities: { [key: string]: string }, keywords: string[], analysis: string, sentiment: { score: number, explanation: string }, topics: string[] } {
      const words = this.tokenize(query);
      const queryVector = this.getTfIdfVector(words);
      
      let bestIntent = '';
      let maxSimilarity = -Infinity;
      
      intents.forEach(intent => {
        const intentVector = this.getTfIdfVector(intent.patterns.join(' ').split(/\s+/));
        const similarity = this.cosineSimilarity(
          Array.from(queryVector.values()),
          Array.from(intentVector.values())
        );
        if (similarity > maxSimilarity) {
          maxSimilarity = similarity;
          bestIntent = intent.patterns[0];
        }
      });
  
      const entities = this.extractEntities(query);
      const keywords = this.extractKeywords(words);
      const sentiment = this.analyzeSentiment(query);
      const topics = this.identifyTopics(query);
  
      let analysis = `Intent: ${bestIntent} (confidence: ${maxSimilarity.toFixed(2)})\n` +
                     `Entities: ${JSON.stringify(entities)}\n` +
                     `Keywords: ${keywords.join(', ')}\n` +
                     `Sentiment: ${sentiment.score.toFixed(2)} - ${sentiment.explanation}\n` +
                     `Topics: ${topics.join(', ')}`;
  
      const contextualAnalysis = this.analyzeContextualRelevance(query);
      analysis += `\nContextual Relevance: ${contextualAnalysis}`;
  
      // Enhanced analysis with additional context
      const additionalContext = this.analyzeAdditionalContext(query);
      analysis += `\nAdditional Context: ${additionalContext}`;
  
      return { intent: bestIntent, entities, keywords, analysis, sentiment, topics };
    }
  
    private analyzeAdditionalContext(query: string): string {
      // Custom logic to provide additional context
      // Example: Analyze query for specific patterns or context clues
      if (query.includes('urgent')) {
        return 'The query seems to be urgent.';
      }
      return 'No additional context identified.';
    }
  
    private getTfIdfVector(words: string[]): Map<string, number> {
      const tf = new Map<string, number>();
      words.forEach(word => {
        tf.set(word, (tf.get(word) || 0) + 1);
      });
  
      const tfidf = new Map<string, number>();
      tf.forEach((freq, word) => {
        const idf = this.idf.get(word) || Math.log(this.documents.length);
        tfidf.set(word, freq * idf);
      });
  
      return tfidf;
    }
  
    private cosineSimilarity(vec1: number[], vec2: number[]): number {
      let dotProduct = 0;
      let magnitudeA = 0;
      let magnitudeB = 0;
      for (let i = 0; i < vec1.length; i++) {
          dotProduct += vec1[i] * vec2[i];
          magnitudeA += vec1[i] ** 2;
          magnitudeB += vec2[i] ** 2;
      }
      magnitudeA = Math.sqrt(magnitudeA);
      magnitudeB = Math.sqrt(magnitudeB);
      if (magnitudeA === 0 || magnitudeB === 0) {
          return 0;
      } else {
          return dotProduct / (magnitudeA * magnitudeB);
      }
  }
  
    private extractEntities(query: string): { [key: string]: string } {
      // Enhanced entity extraction logic
      const entities = this.entityRecognitionModel.recognize(query);
      const additionalEntities = this.extractAdditionalEntities(query);
      return { ...entities, ...additionalEntities };
    }
  
    private extractAdditionalEntities(query: string): { [key: string]: string } {
      // Custom logic to extract additional entities
      const additionalEntities: { [key: string]: string } = {};
      // Example: Extract dates, times, or custom patterns
      const datePattern = /\b\d{4}-\d{2}-\d{2}\b/;
      const match = query.match(datePattern);
      if (match) {
        additionalEntities['date'] = match[0];
      }
      return additionalEntities;
    }
  
    private extractKeywords(words: string[]): string[] {
      const tfidf = this.getTfIdfVector(words);
      return Array.from(tfidf.entries())
        .sort((a, b) => b[1] - a[1])
        .slice(0, 5)
        .map(entry => entry[0]);
    }
  
    private identifyTopics(query: string): string[] {
      return this.topicModel.identify(query);
    }
  
    private generateCoherentResponse(): string {
      const fallbackResponses = [
        "I'm sorry, but I didn't quite understand that. Could you please rephrase?",
        "Can you please provide more details or clarify your question?",
        "I'm here to help! Could you elaborate a bit more on that?"
      ];
  
      return fallbackResponses[Math.floor(Math.random() * fallbackResponses.length)];
    }
  
    learnFromInteraction(query: string, response: string) {
      const normalizedQuery = query.toLowerCase().trim();
      
      if (normalizedQuery === '*yes') {
        // Positive feedback
        this.learningMemory.set(normalizedQuery, { response, feedback: 1 });
        this.trainOnText(query);
        this.trainOnText(response);
        
        // Update knowledge base if the response contains new information
        const potentialNewInfo = response.match(/([^.!?]+[.!?])/g);
        if (potentialNewInfo) {
          potentialNewInfo.forEach(info => {
            const keywords = this.extractKeywords(this.tokenize(info));
            if (keywords.length > 0) {
              const key = keywords.join(' ');
              if (!this.knowledgeBase.has(key)) {
                this.knowledgeBase.set(key, info);
              }
            }
          });
        }
      } else if (normalizedQuery === '*no') {
        // Negative feedback
        this.learningMemory.set(normalizedQuery, { response, feedback: -1 });
        // Here you could implement additional logic to learn from incorrect responses
        // For example, you could flag this response for review or adjust your model
      } else {
        // For other queries, just store them without learning
        this.learningMemory.set(normalizedQuery, { response, feedback: 0 });
      }
    }
  
    updateContext(query: string) {
      this.conversationContext = query;
    }
    generateResponse(intent: string, entities: { [key: string]: string }, keywords: string[], topics: string[],): string {
      let response = '';
  
      // Generate response based on intent
      switch (intent) {
        case 'greeting':
          response = this.generateGreetingResponse();
          break;
        case 'farewell':
          response = this.generateFarewellResponse();
          break;
        case 'thanks':
          response = this.generateThanksResponse();
          break;
        case 'AI':
          response = this.generateAIResponse();
          break;
        default:
          response = this.generateComplexResponse(intent, entities, keywords, topics, this.conversationContext);
      }
  
      // Add relevant information from knowledge base
      const relevantTopics = topics.filter(topic => this.conversationContext.toLowerCase().includes(topic));
      relevantTopics.forEach(topic => {
        if (this.knowledgeBase.has(topic)) {
          const topicInfo = this.knowledgeBase.get(topic)!;
          response += ` ${this.integrateTopicInfo(topicInfo, response)}`;
        }
      });
  
      // Validate and refine the response
      response = this.validateResponse(response);
  
      // Apply sentiment analysis and adjust response tone
  
  
      return response;
    }
  
    private generateGreetingResponse(): string {
      let selectedGreeting;
  
      if (Math.random() < 0.01) {  // 50% chance to use 'greeting' responses
        const greetings = this.aiResponses.get('greeting')!;
        const index = Math.floor(Math.random() * greetings.length);
        selectedGreeting = greetings[index];
      } else {  // 50% chance to use a random response from any category
        const allResponses = Array.from(this.aiResponses.values()).flat();
        const randomIndex = Math.floor(Math.random() * allResponses.length);
        selectedGreeting = allResponses[randomIndex];
      }
  
      const timeOfDay = this.getTimeOfDay();
      let response = `${timeOfDay}! ${selectedGreeting}`;
      
      const hour = new Date().getHours();
      if (hour >= 12 && hour < 18) {
        response += " Well, I think it is afternoon in your world. How's it going?";
      } else if (hour >= 5 && hour < 12) {
        response += " Well, I think it is morning in your world. How's it going?";
      }
      
      return response;
    }
  
    private generateFarewellResponse(): string {
      const farewells = this.aiResponses.get('farewell')!;
      const selectedFarewell = farewells[Math.floor(Math.random() * farewells.length)];
      return `${selectedFarewell} Don't hesitate to return if you have more questions!`;
    }
  
    private generateThanksResponse(): string {
      const thanksResponses = this.aiResponses.get('thanks')!;
      const selectedResponse = thanksResponses[Math.floor(Math.random() * thanksResponses.length)];
      return `${selectedResponse} Is there anything else I can help you with?`;
    }
    private generateAIResponse(): string {
      const aiResponses = this.aiResponses.get('AI')!;
      const selectedResponse = aiResponses[Math.floor(Math.random() * aiResponses.length)];
      return selectedResponse;
    }
    private getTimeOfDay(): string {
      const hour = new Date().getHours();
      if (hour < 12) return "Good morning";
      if (hour < 18) return "Good afternoon";
      return "Good evening";
    }
  
    private integrateTopicInfo(topicInfo: string, existingResponse: string): string {
      // Implement logic to seamlessly integrate topic info into the existing response
      // This could involve summarizing the info, or finding a relevant part to include
      return topicInfo; // Placeholder implementation
    }
  
  
  
    private adjustResponseTone(response: string, sentiment: number): string {
      // Adjust the tone of the response based on the sentiment of the user's input
      if (sentiment < -0.5) {
        return `I understand you might be feeling frustrated. ${response}`;
      } else if (sentiment > 0.5) {
        return `I'm glad you're in a good mood! ${response}`;
      }
      return response;
    }
  
    private isValidResponse(response: string): boolean {
      const unrelatedKeywords = ['python', 'ios', 'hola', 'train', 'subscription', 'privacy'];
      const responseWords = response.toLowerCase().split(/\s+/);
      let unrelatedCount = 0;
  
      for (const word of responseWords) {
        if (unrelatedKeywords.includes(word.replace(/[^\w]/g, ''))) {
          unrelatedCount++;
          if (unrelatedCount > 3) {
            return false;
          }
        }
      }
  
      return true;
    }
    private validateResponse(response: string): string {
      // Grammar Check
      if (!this.grammarCheck(response)) {
        response = this.correctGrammar(response);
      }
  
      // Coherence Check
      if (!this.isCoherent(response)) {
        response = this.generateCoherentResponse();
      }
  
      // Relevance Check
      if (!this.isValidResponse(response)) {
        response = this.generateCoherentResponse();
      }
  
      return response;
    }
  
    /**
     * Generates a comprehensive response based on the detected intent, entities, keywords, topics, and user input.
     * Utilizes advanced NLP techniques to refine and validate the response for coherence and relevance.
     * 
     * @param intent - The detected intent from the user's input.
     * @param entities - A dictionary of extracted entities categorized by their types.
     * @param keywords - A list of keywords extracted from the user input.
     * @param topics - A list of relevant topics identified in the user input.
     * @param userInput - The original input provided by the user.
     * @returns A refined and validated response string.
     */
    private generateComplexResponse(
      intent: string,
      entities: { [key: string]: string },
      keywords: string[],
      topics: string[],
      userInput: string
    ): string {
      let response = '';
  
      if (!this.intents || this.intents.length === 0) {
        console.error("Intents are not initialized or empty.");
        return this.generateCoherentResponse();
      }
  
      // Find the intent that matches the detected intent
      const matchedIntent = this.intents.find((i: Intent) => i.patterns.includes(intent));
  
      if (matchedIntent) {
        // Select the most relevant response based on user input
        response = this.selectBestResponse(matchedIntent.responses, userInput);
      } else {
        // Generate a contextual sentence if no intent matches
        response = this.generateContextualSentence(keywords, userInput);
      }
  
      // Apply advanced NLP refinements to enhance the response
      response = this.applyNLPRefinement(response, intent, entities, keywords, topics, userInput);
  
      // Validate and correct the response for grammar, coherence, and relevance
      response = this.validateResponse(response);
  
      // Log the generated response for debugging and analysis
      console.log("Generated Complex Response:", response);
  
      return response;
    }
  
    /**
     * Selects the most relevant response from a list of possible responses based on the user's input.
     * Utilizes a relevance scoring mechanism to determine the best match.
     * 
     * @param responses - An array of potential response strings.
     * @param userInput - The original input provided by the user.
     * @returns The most relevant response string.
     */
    private selectBestResponse(responses: string[], userInput: string): string {
      if (responses.length === 0) {
        console.warn("No responses available to select from.");
        return this.generateCoherentResponse();
      }
  
      let bestResponse = responses[0];
      let highestRelevance = this.calculateRelevance(bestResponse, userInput);
  
      for (const currentResponse of responses) {
        const currentRelevance = this.calculateRelevance(currentResponse, userInput);
        if (currentRelevance > highestRelevance) {
          bestResponse = currentResponse;
          highestRelevance = currentRelevance;
        }
      }
  
      // Log the selected response and its relevance score
      console.log(`Selected Response: "${bestResponse}" with Relevance: ${highestRelevance.toFixed(2)}`);
  
      return bestResponse;
    }
  
    /**
     * Generates a contextual sentence based on the provided keywords and user input.
     * Enhances the response by incorporating relevant context extracted from the user's message.
     * 
     * @param keywords - A list of keywords extracted from the user input.
     * @param userInput - The original input provided by the user.
     * @returns A contextual sentence string.
     */
    private generateContextualSentence(keywords: string[], userInput: string): string {
      if (keywords.length === 0) {
        console.warn("No keywords provided for generating a contextual sentence.");
        return this.generateCoherentResponse();
      }
  
      const primaryKeyword = keywords[0];
      const context = this.extractContext(userInput);
  
      // Generate a sentence that relates the primary keyword with the extracted context
      const sentence = this.generateSentence(primaryKeyword, context, 20);
  
      // Log the generated contextual sentence
      console.log(`Generated Contextual Sentence: "${sentence}"`);
  
      return sentence;
    }
  
    private applyNLPRefinement(response: string, intent: string, entities: { [key: string]: string }, keywords: string[], topics: string[], userInput: string): string {
      // Apply various NLP techniques like named entity recognition, coreference resolution, etc.
      // This is a placeholder for the actual implementation
      return response;
    }
  
    private getRecentContextHistory(): string[] {
      // Retrieve recent conversation context
      // This is a placeholder for the actual implementation
      return [];
    }
  
    private integrateResponses(original: string, refined: string, improved: string): string {
      // Coherently combine the responses
      // This is a placeholder for the actual implementation
      return `${original} ${refined} ${improved}`;
    }
  
    private applyFinalTouches(response: string, userInput: string): string {
      // Apply final refinements for naturalness and coherence
      // This is a placeholder for the actual implementation
      return response;
    }
  
    private calculateRelevance(response: string, userInput: string): number {
      // Calculate the relevance of the response to the user input
      // This is a placeholder for the actual implementation
      return 0;
    }
  
    private extractContext(userInput: string): string {
      // Extract relevant context from the user input
      // This is a placeholder for the actual implementation
      return userInput;
    }
  
    findMostSimilarIntent(query: string, intents: Intent[]): Intent | null {
      const { intent } = this.understandQuery(query);
      return intents.find(i => i.patterns.includes(intent)) || null;
    }
  
    private findSimilarWords(word: string, n: number): string[] {
      if (!this.wordVectors.has(word)) return [];
  
      const wordVector = this.wordVectors.get(word)!;
      const similarities = Array.from(this.wordVectors.entries())
        .map(([w, vec]) => [w, this.cosineSimilarity(vec, wordVector)])
        .sort((a, b) => (b[1] as number) - (a[1] as number))
        .slice(1, n + 1);  // Exclude the word itself
  
      return similarities.map(s => s[0] as string);
    }
  
  
  
    private generateContextFromAnalysis(analysis: ReturnType<typeof this.understandQuery>): string {
      return `${analysis.intent} ${analysis.keywords.join(' ')} ${Object.values(analysis.entities).join(' ')}`;
    }
  
    private analyzeContextualRelevance(query: string): string {
      const queryVector = this.getTfIdfVector(this.tokenize(query));
      const contextVector = this.getTfIdfVector(this.tokenize(this.contextMemory.join(' ')));
      const similarity = this.cosineSimilarity(Array.from(queryVector.values()), Array.from(contextVector.values()));
      return `Similarity to context: ${similarity.toFixed(2)}`;
    }
    private updateWordProbabilities(sentence: string) {
      const words = this.tokenize(sentence);
      const decayFactor = 0.9;
      for (let i = 0; i < words.length - 1; i++) {
        const currentWord = words[i];
        const nextWord = words[i + 1];
        if (!this.wordProbabilities.has(currentWord)) {
          this.wordProbabilities.set(currentWord, new Map());
        }
        const nextWordProbs = this.wordProbabilities.get(currentWord)!;
        const currentProb = nextWordProbs.get(nextWord) || 0;
        const newProb = currentProb * decayFactor + (1 - decayFactor);
        nextWordProbs.set(nextWord, newProb);
      }
    }
  
    private getNextWordProbability(currentWord: string, nextWord: string): number {
      if (!this.wordProbabilities.has(currentWord)) return 0;
      const nextWordProbs = this.wordProbabilities.get(currentWord)!;
      const totalOccurrences = Array.from(nextWordProbs.values()).reduce((sum, count) => sum + count, 0);
      return (nextWordProbs.get(nextWord) || 0) / totalOccurrences;
    }
  
    generateComplexSentence(startWord: string, userInput: string, maxLength: number = 40): string {
      let sentence = [startWord];
      let currentContext = userInput;
      let wordCount = 1;
      let topicStack: string[] = [];
      let sentimentHistory: number[] = [];
  
      // Initialize topic and sentiment
      const initialAnalysis = this.analyzeContext(currentContext);
      topicStack.push(...initialAnalysis.topics);
      sentimentHistory.push(initialAnalysis.sentiment.score);
  
      while (wordCount < maxLength) {
        // Combine current sentence with original user input for context
        const combinedContext = `${userInput} ${sentence.join(' ')}`;
        
        // Encode combined context
        const meaningVector = this.encodeToMeaningSpace(combinedContext);
        
        // Apply GAN refinement
        const refinedVector = this.gan.refine(meaningVector, wordCount);
        
        // Apply RL improvement
        const improvedVector = this.rlAgent.improve(refinedVector, {
          topicStack,
          sentimentHistory,
          wordCount,
          maxLength,
          userInput,
          previousWords: sentence
        });
        
        // Predict next word
        const nextWordVector = this.decoder.predict(improvedVector);
        const nextWord = this.findClosestWord(nextWordVector);
        
        // Enforce topic adherence and coherence
        const topicAdherentWord = this.enforceTopicAdherence(nextWord, topicStack[topicStack.length - 1]);
        if (!this.isCoherent(sentence.join(' ') + ' ' + topicAdherentWord)) {
          continue; // Skip this word and try again
        }
        
        // Analyze context including the potential next word
        const { sentiment, topics } = this.analyzeContext(`${combinedContext} ${topicAdherentWord}`);
        
        // Adjust word based on analysis and ensure naturalness
        const adjustedNextWord = this.adjustWordForNaturalness(topicAdherentWord, sentiment, topics, sentence);
        
        // Add word to sentence
        sentence.push(adjustedNextWord);
        currentContext = `${userInput} ${sentence.join(' ')}`;
        wordCount++;
  
        // Update topic stack and sentiment history
        if (topics.length > 0 && topics[0] !== topicStack[topicStack.length - 1]) {
          topicStack.push(topics[0]);
        }
        sentimentHistory.push(sentiment.score);
  
        // Check for sentence end
        if (this.shouldEndSentence(adjustedNextWord, wordCount, maxLength, topicStack, sentimentHistory, userInput)) {
          break;
        }
  
        // Update word probabilities
        this.updateWordProbabilities(currentContext);
        
        // Periodic context refresh
        if (wordCount % 5 === 0) {
          currentContext = this.refreshContext(sentence, userInput);
        }
      }
  
      return this.postProcessSentence(sentence.join(' '), userInput);
    }
  
    private adjustWordForNaturalness(word: string, sentiment: any, topics: string[], previousWords: string[]): string {
      const commonPhrases = [
        "How can I help you",
        "Check out our latest",
        "Feel free to explore",
        "Let me know if you have any questions",
        "I'm here to assist you"
      ];
  
      for (const phrase of commonPhrases) {
        const phraseWords = phrase.toLowerCase().split(' ');
        const lastWordIndex = previousWords.length - 1;
        if (lastWordIndex >= 0 && phraseWords.includes(previousWords[lastWordIndex].toLowerCase())) {
          const nextWordInPhrase = phraseWords[phraseWords.indexOf(previousWords[lastWordIndex].toLowerCase()) + 1];
          if (nextWordInPhrase) {
            return nextWordInPhrase;
          }
        }
      }
  
      // If no common phrase is applicable, use the original word
      return word;
    }
  
    private postProcessSentence(sentence: string, userInput: string): string {
      // Capitalize the first letter of the sentence
      sentence = sentence.charAt(0).toUpperCase() + sentence.slice(1);
      
      // Ensure the sentence ends with proper punctuation
      if (!sentence.endsWith('.') && !sentence.endsWith('!') && !sentence.endsWith('?')) {
        sentence += '.';
      }
      
      // Add a friendly greeting if it's missing
      if (!sentence.toLowerCase().startsWith('hello') && !sentence.toLowerCase().startsWith('hi')) {
        const greetings = ["Hi", "Hello", "Hey", "Greetings", "Good day"];
        sentence = greetings[Math.floor(Math.random() * greetings.length)] + " " + sentence;
      }
      
      // Ensure the sentence includes a reference to helping the user
      if (!sentence.toLowerCase().includes('help you')) {
        sentence += ["Is there anything I can assist you with?", "What can I do for you?", "How may I be of service?", "What brings you here today?", "In what way can I offer my help?"][Math.floor(Math.random() * 5)];
      }
      
      // Add a reference to MazsAI if it's not already included
      if (!sentence.includes('MazsAI')) {
        sentence += ' Feel free to check out our latest model at MazsAI.';
      }
      
      return sentence;
    }
  
    private isCoherent(sentence: string): boolean {
      const words = sentence.split(/\s+/);
      const wordSet = new Set<string>();
      let repetitionCount = 0;
  
      for (const word of words) {
        const lowerWord = word.toLowerCase();
        if (wordSet.has(lowerWord)) {
          repetitionCount++;
          if (repetitionCount > 2) {
            return false; // More than two repetitions indicate incoherence
          }
        } else {
          wordSet.add(lowerWord);
        }
      }
      return true;
    }
  
    private findRelatedWordFromKnowledgeBase(word: string, previousWord: string): string | null {
      for (const [, value] of Array.from(this.knowledgeBase.entries())) {
        if (value.includes(word) || value.includes(previousWord)) {
          const relatedWords = value.split(' ').filter((w: string) => w !== word && w !== previousWord);
          if (relatedWords.length > 0) {
            return relatedWords[Math.floor(Math.random() * relatedWords.length)];
          }
        }
      }
      return null;
    }
  
    private refreshContext(sentence: string[], userInput: string): string {
      // Implement logic to refresh the context, incorporating the original user input
      return `${userInput} ${sentence.slice(-10).join(' ')}`;
    }
  
  
  
    private shouldEndSentence(word: string, currentLength: number, maxLength: number, topicStack: string[], sentimentHistory: number[], userInput: string): boolean {
      // Enhanced logic for ending the sentence, considering user input
      if (word.endsWith('.') || word.endsWith('!') || word.endsWith('?')) {
        return true;
      }
  
      if (currentLength >= maxLength - 5) {
        return true;
      }
  
      // End if topic has changed multiple times
      if (topicStack.length > 3) {
        return true;
      }
  
      // End if sentiment has fluctuated significantly
      if (sentimentHistory.length > 5) {
        const recentSentiments = sentimentHistory.slice(-5);
        const sentimentVariance = this.calculateVariance(recentSentiments);
        if (sentimentVariance > 0.5) {
          return true;
        }
      }
  
      // End if the sentence has covered the main points of the user input
      const userInputKeywords = this.extractKeywords(userInput.split(' '));
      const sentenceKeywords = this.extractKeywords([word]);
      if (userInputKeywords.every(keyword => sentenceKeywords.includes(keyword))) {
        return true;
      }
  
      const endProbability = Math.min(0.1, currentLength / maxLength);
      return Math.random() < endProbability;
    }
  
    private calculateVariance(numbers: number[]): number {
      let mean = 0;
      let M2 = 0;
      let n = 0;
  
      numbers.forEach(num => {
        n += 1;
        const delta = num - mean;
        mean += delta / n;
        M2 += delta * (num - mean);
      });
  
      return n > 1 ? M2 / (n - 1) : 0; // Use n - 1 for sample variance
    }
  
    updateConversationHistory(role: 'user' | 'ai', content: string) {
      this.conversationHistory.push({ role, content });
      if (this.conversationHistory.length > 10) {
        this.conversationHistory.shift();
      }
    }
  
    recognizeEntities(text: string): { [key: string]: string[] } {
      const entities: { [key: string]: string[] } = {
        person: [],
        organization: [],
        location: [],
        date: [],
      };
  
      // Simple pattern matching for entity recognition
      const words = text.split(' ');
      words.forEach(word => {
        if (/^[A-Z][a-z]+$/.test(word)) {
          entities.person.push(word);
        }
        if (/^[A-Z]{2,}$/.test(word)) {
          entities.organization.push(word);
        }
        if (/^[A-Z][a-z]+(?:,\s[A-Z]{2})?$/.test(word)) {
          entities.location.push(word);
        }
        if (/^\d{1,2}\/\d{1,2}\/\d{2,4}$/.test(word)) {
          entities.date.push(word);
        }
      });
  
      return entities;
    }
  
    translateText(text: string, targetLanguage: string): string {
      const translations: { [key: string]: { [key: string]: string } } = {
        'hello': { 'es': 'hola', 'fr': 'bonjour', 'de': 'hallo' },
        'goodbye': { 'es': 'adiós', 'fr': 'au revoir', 'de': 'auf wiedersehen' },
        'how are you': { 'es': 'cómo estás', 'fr': 'comment allez-vous', 'de': 'wie geht es dir' },
        // Add more translations as needed
        'thank you': { 'es': 'gracias', 'fr': 'merci', 'de': 'danke' },
        'please': { 'es': 'por favor', 'fr': 's\'il vous plaît', 'de': 'bitte' },
        'yes': { 'es': 'sí', 'fr': 'oui', 'de': 'ja' },
        'no': { 'es': 'no', 'fr': 'non', 'de': 'nein' },
        'good morning': { 'es': 'buenos días', 'fr': 'bonjour', 'de': 'guten morgen' },
        'good night': { 'es': 'buenas noches', 'fr': 'bonne nuit', 'de': 'gute nacht' },
        // Add more translations as needed
      };
  
      return text.split(' ').map(word => {
        const lowerWord = word.toLowerCase();
        return translations[lowerWord]?.[targetLanguage] || word;
      }).join(' ');
    }
  
  
    private buildMarkovChain(text: string) {
      // Split the text into sentences using a regex that matches sentence-ending punctuation
      const sentences = text.match(/[^.!?]+[.!?]+/g) || [];
      
      sentences.forEach(sentence => {
        // Tokenize each sentence into words, handling punctuation and contractions
        const words = this.enhancedTokenize(sentence);
        // Add start and end tokens to capture sentence boundaries
        const tokens = ['<START>', ...words, '<END>'];
        
        // Iterate through the tokens to build quadgrams for more context
        for (let i = 0; i < tokens.length - 3; i++) {
          const currentContext = `${tokens[i]} ${tokens[i + 1]}`;
          const nextWord = tokens[i + 2];
          const nextNextWord = tokens[i + 3];
          
          // Initialize the map for the current context if it doesn't exist
          if (!this.markovChain.has(currentContext)) {
            this.markovChain.set(currentContext, new Map());
          }
          const contextMap = this.markovChain.get(currentContext)!;
          
          // Update the frequency count for the next two words
          const nextPair = `${nextWord} ${nextNextWord}`;
          contextMap.set(nextPair, (contextMap.get(nextPair) || 0) + 1);
        }
      });
      
      // Smooth the Markov chain to handle unseen quadgrams
      this.smoothMarkovChain();
    }
  
    private enhancedTokenize(sentence: string): string[] {
      // Use regex to split words while keeping punctuation attached
      return sentence
        .toLowerCase()
        .match(/\b\w+'?\w*\b|[.!?,;]/g) || [];
    }
  
    private smoothMarkovChain() {
      const smoothingFactor = 0.5;
      this.markovChain.forEach((nextPairs, context) => {
        let total = 0;
        // Apply smoothing to each next pair
        nextPairs.forEach((count, pair) => {
          nextPairs.set(pair, count + smoothingFactor);
          total += count + smoothingFactor;
        });
        
        // Add a smoothing term for unseen pairs
        // Assuming a fixed number of possible pairs; alternatively, calculate based on vocabulary
        const uniquePairs = this.getUniquePairs();
        const unseenPairs = uniquePairs.length - nextPairs.size;
        total += unseenPairs * smoothingFactor;
        
        // Normalize the frequencies to probabilities
        nextPairs.forEach((count, pair) => {
          nextPairs.set(pair, count / total);
        });
      });
    }
  
    private getUniquePairs(): string[] {
      const pairs = new Set<string>();
      this.markovChain.forEach((nextPairs) => {
        nextPairs.forEach((_, pair) => pairs.add(pair));
      });
      return Array.from(pairs);
    }
  
    private generateTextUsingMarkovChain(startContext: string, maxLength: number = 30): string {
      let [currentWord, nextWord] = startContext.split(' ');
      let sentence = [currentWord, nextWord];
  
      for (let i = 2; i < maxLength; i++) {
        const currentContext = `${currentWord} ${nextWord}`;
        const nextPairs = this.markovChain.get(currentContext);
        if (!nextPairs) break;
  
        const totalFrequency = Array.from(nextPairs.values()).reduce((sum, freq) => sum + freq, 0);
        let random = Math.random() * totalFrequency;
            // Start of Selection
            let selectedPair = '';
  
            const entries = Array.from(nextPairs.entries());
            for (let i = 0; i < entries.length; i++) {
              const [pair, freq] = entries[i];
              random -= freq;
              if (random <= 0) {
                selectedPair = pair;
                break;
              }
            }
  
  
        if (!selectedPair) break;
  
        const [newWord] = selectedPair.split(' ');
        if (newWord === '<END>') break;
  
        sentence.push(newWord);
        currentWord = nextWord;
        nextWord = newWord;
  
        if (newWord.endsWith('.') || newWord.endsWith('!') || newWord.endsWith('?')) break;
      }
  
      return sentence.join(' ');
    }}
  
  // Advanced sentiment analysis model
  class AdvancedSentimentModel {
    analyze(text: string): { score: number, explanation: string } {
      // Implement a more sophisticated sentiment analysis algorithm
      const positiveWords = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'happy', 'joy', 'love', 'like', 'best'];
      const negativeWords = ['bad', 'terrible', 'awful', 'horrible', 'disappointing', 'sad', 'angry', 'hate', 'dislike', 'worst'];
      let score = 0;
      text.toLowerCase().split(/\s+/).forEach(word => {
        if (positiveWords.includes(word)) score++;
        if (negativeWords.includes(word)) score--;
      });
      const explanation = `Sentiment score: ${score}`;
      return { score, explanation };
    }
  }
  
  // Entity recognition model
  class EntityRecognitionModel {
    recognize(text: string): { [key: string]: string } {
      // Implement a more robust entity recognition algorithm
      const entities: { [key: string]: string } = {};
      const dateMatch = text.match(/\b(\d{4}-\d{2}-\d{2}|\d{1,2}\/\d{1,2}\/\d{2,4})\b/);
      if (dateMatch) entities['date'] = dateMatch[0];
      const nameMatch = text.match(/\b([A-Z][a-z]+ [A-Z][a-z]+)\b/);
      if (nameMatch) entities['name'] = nameMatch[0];
      const emailMatch = text.match(/\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b/);
      if (emailMatch) entities['email'] = emailMatch[0];
      const locationMatch = text.match(/\b([A-Z][a-z]+(?: [A-Z][a-z]+)*)\b/);
      if (locationMatch) entities['location'] = locationMatch[0];
      return entities;
    }
  }
  
  // Topic modeling
  class TopicModel {
    identify(text: string): string[] {
      // Implement a topic modeling algorithm
      // This is a placeholder implementation
      const topics = ['ai', 'machine learning', 'deep learning'];
      return topics.filter(topic => text.toLowerCase().includes(topic));
    }
  }
  
  class GAN {
    private generator: MultilayerPerceptron;
    private discriminator: MultilayerPerceptron;
    private latentDim: number = 100;
    private realData: number[][];
    wordVectors: any;
    getContextWords: any;
  
    constructor(realData: number[][]) {
      this.realData = realData;
      this.generator = new MultilayerPerceptron(
        [this.latentDim, 128, 256, 128, 100],
        ['relu', 'relu', 'relu', 'tanh']
      );
      this.discriminator = new MultilayerPerceptron(
        [100, 128, 256, 128, 1],
        ['relu', 'relu', 'relu', 'sigmoid']
      );
    }
  
    refine(meaningVector: number[], sentenceLength: number): number[] {
      const noiseFactor = Math.max(0.1, 1 - sentenceLength / 100);
      const noise = Array.from({ length: this.latentDim }, () => Math.random() * 2 - 1).map(n => n * noiseFactor);
      const generatedVector = this.generator.predict([...noise, ...meaningVector]);
      return generatedVector;
    }
  
    generateText(latentVector: number[]): string {
      const outputVector = this.generator.predict(latentVector);
      return this.vectorToText(outputVector);
    }
  
    private vectorToText(action: number[], options: {
      mode: 'basic' | 'advanced' | 'semantic';
      threshold?: number;
      semanticDictionary?: Map<string, number[]>;
    } = { mode: 'basic' }): string {
      const basicDictionary: string[] = [
        'alpha', 'bravo', 'charlie', 'delta', 'echo', 'foxtrot', 'golf',
        'hotel', 'india', 'juliet', 'kilo', 'lima', 'mike', 'november',
        'oscar', 'papa', 'quebec', 'romeo', 'sierra', 'tango', 'uniform',
        'victor', 'whiskey', 'xray', 'yankee', 'zulu'
      ];
  
      const advancedDictionary: string[] = [
        ...basicDictionary,
        'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine',
        'plus', 'minus', 'multiply', 'divide', 'equals', 'percent',
        'north', 'south', 'east', 'west', 'up', 'down', 'left', 'right',
        'start', 'end', 'pause', 'continue', 'stop', 'go'
      ];
  
      switch (options.mode) {
        case 'basic':
          return action.map(num => {
            const index = Math.abs(Math.round(num)) % basicDictionary.length;
            return basicDictionary[index];
          }).join(' ');
  
        case 'advanced':
          const threshold = options.threshold || 0.5;
          return action.map(num => {
            if (Math.abs(num) < threshold) return '';
            const index = Math.abs(Math.round(num)) % advancedDictionary.length;
            return (num < 0 ? 'not_' : '') + advancedDictionary[index];
          }).filter(word => word !== '').join(' ');
  
        case 'semantic':
          if (!options.semanticDictionary) {
            throw new Error('Semantic dictionary is required for semantic mode');
          }
          return action.map(num => {
            let closestWord = '';
            let minDistance = Infinity;
            Array.from(options.semanticDictionary!.entries()).forEach(([word, vector]) => {
              const distance = this.euclideanDistance(action, vector);
              if (distance < minDistance) {
                minDistance = distance;
                closestWord = word;
              }
            });
            return closestWord;
          }).join(' ');
  
        default:
          throw new Error('Invalid mode specified');
      }
    }
  
    private euclideanDistance(vec1: number[], vec2: number[]): number {
      if (vec1.length !== vec2.length) {
        throw new Error('Vectors must have the same length');
      }
      return Math.sqrt(vec1.reduce((sum, val, i) => sum + Math.pow(val - vec2[i], 2), 0));
    }
  
    train(realData: number[][], epochs: number = 10, batchSize: number = 64) {
      for (let epoch = 0; epoch < epochs; epoch++) {
        // Train discriminator
        const realBatch = this.getBatch(realData, batchSize);
        const fakeBatch = this.generateFakeBatch(batchSize);
        
        realBatch.forEach(real => {
          this.discriminator.train(real, [1], 0.0002);
        });
        
        fakeBatch.forEach(fake => {
          this.discriminator.train(fake, [0], 0.0002);
        });
  
        // Train generator
        const noise = this.generateNoise(batchSize);
        noise.forEach(n => {
          const fake = this.generator.predict(n);
          const target = this.discriminator.predict(fake).map(d => 1 - d); // Target is 1 for generator
          this.generator.train(n, target, 0.0002);
        });
  
        // Log progress and losses
        if (epoch % 10 === 0) {
          const gLoss = this.generatorLoss();
          const dLoss = this.discriminatorLoss();
          console.log(`GAN Epoch ${epoch}: G Loss: ${gLoss}, D Loss: ${dLoss}`);
        }
  
        // Adaptive learning rate adjustment
        if (epoch % 50 === 0 && epoch > 0) {
          this.adjustLearningRates(epoch);
        }
      }
    }
  
    private adjustLearningRates(epoch: number) {
      const newLearningRate = 0.0002 * Math.pow(0.95, Math.floor(epoch / 50));
      this.generator.setLearningRate(newLearningRate);
      this.discriminator.setLearningRate(newLearningRate);
      console.log(`Adjusted learning rates to ${newLearningRate}`);
    }
  
    private getBatch(data: number[][], batchSize: number): number[][] {
      const batch = [];
      for (let i = 0; i < batchSize; i++) {
        const index = Math.floor(Math.random() * data.length);
        batch.push(data[index]);
      }
      return batch;
    }
  
    private generateFakeBatch(batchSize: number): number[][] {
      return this.generateNoise(batchSize).map(noise => this.generator.predict(noise));
    }
  
    private generateNoise(batchSize: number): number[][] {
      return Array.from({ length: batchSize }, () => 
        Array.from({ length: this.latentDim }, () => Math.random() * 2 - 1)
      );
    }
  
    private generatorLoss(): number {
      const fakeBatch = this.generateFakeBatch(32);
      return fakeBatch.reduce((loss, fake) => {
        const discriminatorOutput = this.discriminator.predict(fake)[0];
        return loss - Math.log(discriminatorOutput);
      }, 0) / 32;
    }
  
    private discriminatorLoss(): number {
      const realBatch = this.getBatch(this.realData, 32);
      const fakeBatch = this.generateFakeBatch(32);
      
      const realLoss = realBatch.reduce((loss, real) => {
        const discriminatorOutput = this.discriminator.predict(real)[0];
        return loss - Math.log(discriminatorOutput);
      }, 0) / 32;
  
      const fakeLoss = fakeBatch.reduce((loss, fake) => {
        const discriminatorOutput = this.discriminator.predict(fake)[0];
        return loss - Math.log(1 - discriminatorOutput);
      }, 0) / 32;
  
      return (realLoss + fakeLoss) / 2;
    }
  }
  
  class RLAgent {
    private policy: MultilayerPerceptron;
    private valueNetwork: MultilayerPerceptron;
    private gamma: number = 0.99;
    private epsilon: number = 0.1;
  
    constructor() {
      this.policy = new MultilayerPerceptron([64, 128, 64], ['relu', 'relu']);
      this.valueNetwork = new MultilayerPerceptron([64, 128, 1], ['relu']);
    }
  
    improve(state: number[], context: any): number[] {
      if (Math.random() < this.epsilon) {
        return state.map(() => Math.random() * 2 - 1);
      } else {
        const action = this.policy.predict(state);
        const reward = this.calculateReward(action, context);
        this.train([{ state, action, reward, nextState: action }]);
        return action;
      }
    }
  
    private calculateReward(action: number[], context: any): number {
      const coherence = this.assessCoherence(action, context.previousWords);
      const topicRelevance = this.assessTopicRelevance(action, context.topicStack);
      const sentimentAlignment = this.assessSentimentAlignment(action, context.sentimentHistory);
      
      return coherence * 0.4 + topicRelevance * 0.4 + sentimentAlignment * 0.2;
    }
  
    private assessCoherence(action: number[], previousWords: string[]): number {
      /**
       * Enhanced Coherence Assessment
       * 
       * This function evaluates the coherence of the current action in the context
       * of previous words. It utilizes semantic similarity measures to determine
       * how well the action aligns with the preceding context.
       */
  
      if (!previousWords || !Array.isArray(previousWords)) {
        return 0; // Or some default value
      }
      
      // Convert previous words array to a single string
      const context = previousWords.join(' ');
  
      // Convert action vector to a meaningful representation
      const actionText = this.vectorToText(action);
  
      // Calculate semantic similarity between action and context
      const similarityScore = this.calculateSemanticSimilarity(actionText, context);
  
      // Normalize the similarity score to a range between 0 and 1
      const normalizedScore = (similarityScore + 1) / 2;
  
      // Ensure the score is within bounds
      return Math.max(0, Math.min(normalizedScore, 1));
    }
  
    /**
     * Converts an action vector to its corresponding text representation.
     * This is a placeholder for actual implementation.
     * @param action - The action represented as a number array.
     * @returns The textual representation of the action.
     */
    private vectorToText(action: number[]): string {
      // Convert the action vector into meaningful text using a predefined dictionary
      const dictionary: string[] = [
        'alpha', 'bravo', 'charlie', 'delta', 'echo', 'foxtrot', 'golf',
        'hotel', 'india', 'juliet', 'kilo', 'lima', 'mike', 'november',
        'oscar', 'papa', 'quebec', 'romeo', 'sierra', 'tango', 'uniform',
        'victor', 'whiskey', 'xray', 'yankee', 'zulu'
      ];
  
      // Map each number in the action vector to a corresponding word in the dictionary
      return action.map(num => {
        const index = num % dictionary.length;
        return dictionary[index];
      }).join(' ');
    }
  
    /**
     * Calculates the semantic similarity between two text strings.
     * This is a placeholder for actual implementation using NLP techniques.
     * @param text1 - The first text string.
     * @param text2 - The second text string.
     * @returns A similarity score between -1 and 1.
     */
    private calculateSemanticSimilarity(text1: string, text2: string): number {
      /**
       * Enhanced implementation using Cosine Similarity.
       * This method provides a more accurate semantic similarity score between two texts
       * by considering the frequency of each term and calculating the cosine of the angle
       * between their term frequency vectors.
       */
  
      // Helper function to tokenize and clean the text
      const tokenize = (text: string): string[] => {
        return text
          .toLowerCase()
          .replace(/[^\w\s]/g, '') // Remove punctuation
          .split(/\s+/) // Split by whitespace
          .filter(word => word.length > 0); // Remove empty strings
      };
  
      // Helper function to create a term frequency map
      const termFrequency = (tokens: string[]): Map<string, number> => {
        const frequencyMap = new Map<string, number>();
        tokens.forEach(token => {
          frequencyMap.set(token, (frequencyMap.get(token) || 0) + 1);
        });
        return frequencyMap;
      };
  
      // Tokenize both texts
      const tokens1 = tokenize(text1);
      const tokens2 = tokenize(text2);
  
      // Create term frequency maps
      const tf1 = termFrequency(tokens1);
      const tf2 = termFrequency(tokens2);
  
      // Create a set of all unique terms from both texts
      const allTerms = new Set<string>();
      tf1.forEach((_, key) => allTerms.add(key));
      tf2.forEach((_, key) => allTerms.add(key));
  
      // Initialize term frequency vectors
      const vector1: number[] = [];
      const vector2: number[] = [];
  
      // Populate the vectors
      allTerms.forEach(term => {
        vector1.push(tf1.get(term) || 0);
        vector2.push(tf2.get(term) || 0);
      });
  
      // Calculate the dot product of the vectors
      const dotProduct = vector1.reduce((sum, val, idx) => sum + val * vector2[idx], 0);
  
      // Calculate the magnitude of each vector
      const magnitude1 = Math.sqrt(vector1.reduce((sum, val) => sum + val * val, 0));
      const magnitude2 = Math.sqrt(vector2.reduce((sum, val) => sum + val * val, 0));
  
      // Handle division by zero
      if (magnitude1 === 0 || magnitude2 === 0) {
        return 0; // No similarity if one of the vectors is zero
      }
  
      // Calculate Cosine Similarity
      const cosineSimilarity = dotProduct / (magnitude1 * magnitude2);
  
      // Clamp the similarity score to ensure it falls within the range [-1, 1]
      return Math.max(-1, Math.min(1, cosineSimilarity));
      }
  
    private assessTopicRelevance(action: number[], topicStack: string[]): number {
      // Implement topic relevance assessment
      // This is a placeholder implementation
      return Math.random();
    }
  
    private assessSentimentAlignment(action: number[], sentimentHistory: number[]): number {
      // Implement sentiment alignment assessment
      // This is a placeholder implementation
      return Math.random();
    }
       
    train(experiences: { state: number[], action: number[], reward: number, nextState: number[] }[]) {
      experiences.forEach(exp => {
        const targetValue = exp.reward + this.gamma * this.valueNetwork.predict(exp.nextState)[0];
        const currentValue = this.valueNetwork.predict(exp.state)[0];
        const advantage = targetValue - currentValue;
  
        this.valueNetwork.train(exp.state, [targetValue], 0.001);
        this.policy.train(exp.state, exp.action.map(a => a * advantage), 0.001);
      });
    }
  }
  
      // Start of Selection
      function solvePuzzle(puzzle: string): string {
        const puzzles = new Map([
          ['if you have a bee in your hand what do you have in your eye', {
            answer: 'beauty',
            explanation: 'This is a play on words. "Beauty" sounds like "bee-uty", combining "bee" and "eye".'
          }],
          ['what has keys but no locks a space but no room and you can enter but not go in', {
            answer: 'keyboard',
            explanation: 'A keyboard has keys but no locks, a space bar but no physical room, and you can enter (press the Enter key) but not physically go inside it.'
          }],
          ['what comes once in a minute twice in a moment but never in a thousand years', {
            answer: 'the letter M',
            explanation: 'The letter "M" appears once in "minute", twice in "moment", and doesn\'t appear in "thousand years".'
          }],
          // Add more puzzles here
        ]);
      
        const normalizedPuzzle = puzzle.toLowerCase().replace(/[^a-z ]/g, '');
        const puzzleSolution = puzzles.get(normalizedPuzzle);
      
        if (puzzleSolution) {
          return `
      Puzzle: "${puzzle}"
      
      Solution: The answer is "${puzzleSolution.answer}".
      
      Explanation:
      ${puzzleSolution.explanation}
      
      Solving Process:
      1. Identify the key elements of the puzzle:
         - "bee in your hand"
         - "in your eye"
      2. Look for potential wordplay or hidden meanings.
      3. Connect the elements to form a logical answer.
      4. Verify the answer fits all parts of the puzzle.
      
      Is there another puzzle you'd like me to solve?`;
        } else {
          return `I'm sorry, but I don't have a specific solution for that puzzle in my database. However, I can offer some general tips for solving puzzles:
      
      1. Read the puzzle carefully and identify key words or phrases.
      2. Look for potential wordplay, puns, or double meanings.
      3. Think about common sayings or idioms that might relate to the puzzle.
      4. Consider different perspectives or interpretations of the puzzle elements.
      5. Don't be afraid to think outside the box!
      
      If you have a different puzzle, feel free to ask and I'll do my best to help!`;
        }
      }
  
      export function processChatbotQuery(query: string): string {
        // Analyze the user query to extract intent, entities, keywords, analysis metrics, sentiment, and topics
        const { intent, entities, keywords, analysis, sentiment, topics } = nlp.understandQuery(query);
        console.log("Query Analysis:", analysis);
  
        // Update the conversation history with the user's query
        nlp.updateConversationHistory('user', query);
  
        // Recognize and extract entities present in the user's query
        const recognizedEntities = nlp.recognizeEntities(query);
        console.log("Recognized Entities:", recognizedEntities);
  
        // Extract the confidence score from the analysis using regex
        const confidenceMatch = analysis.match(/confidence:\s*([\d.]+)/);
        const confidence = confidenceMatch ? Math.min(Math.max(parseFloat(confidenceMatch[1]), 0), 1) : 0;
        if (isNaN(confidence)) {
          console.warn("Invalid confidence value detected. Defaulting to 0.");     
        }
        console.log("Confidence Score:", confidence);
  
        // Check if the query is a puzzle
        if (query.toLowerCase().includes('puzzle') || query.toLowerCase().includes('riddle')) {
          return solvePuzzle(query);
        }
  
        // Check if it's a math calculation
        if (query.match(/[0-9+\-*/%^()]/)) {
          return generateComplexMathCalculation(query);
        }
  
        // If the confidence score is below the threshold, generate an uncertain response
        if (confidence < 1) {
          const uncertainResponse = nlp.generateComplexSentence("I'm not sure I understand", "uncertain response", 20);
          console.log("Uncertain Response Generated:", uncertainResponse);
          nlp.updateConversationHistory('ai', uncertainResponse);
          return uncertainResponse;
        }
  
        // Find the intent that matches the extracted intent from the query
        const matchedIntent = intents.find(i => i.patterns.includes(intent));
        if (matchedIntent) {
          // Generate an initial response based on the matched intent and extracted data
          let response = nlp.generateResponse(intent, entities, keywords, topics);
          console.log("Initial Response:", response);
          
          // If the intent is not a simple greeting or farewell, generate an additional context sentence
          if (!['hello', 'hi', 'hey', 'bye', 'goodbye', 'see you'].includes(intent)) {
            const primaryKeyword = keywords[0] || response.split(' ')[0];
            const contextSentence = nlp.generateComplexSentence(primaryKeyword, query, 500);
            response += " " + contextSentence;
            console.log("Context Sentence Added:", contextSentence);
          }
  
          // Encode the response into a meaningful vector space using GAN for refinement
          const responseVector = nlp.encodeToMeaningSpace(response);
          const refinedVector = nlp.gan.refine(responseVector, response.split(' ').length);
          const refinedResponse = nlp.decodeFromMeaningSpace(refinedVector);
          console.log("Refined Response:", refinedResponse);
  
          // Use Reinforcement Learning agent to further improve the refined response
          const improvedVector = nlp.rlAgent.improve(refinedVector, {
            intent,
            entities,
            keywords,
            sentiment,
            topics
          });
          const improvedResponse = nlp.decodeFromMeaningSpace(improvedVector);
          console.log("Improved Response:", improvedResponse);
  
          // Combine the original, refined, and improved responses for a comprehensive reply
          response = Math.random() < 0.1 ? ` ${response} \n \n \n  Experimental AI model: ${improvedResponse}` : `${response}`;
          console.log("Combined Response:", response);
  
          // Analyze the sentiment of the query to adjust the response accordingly
          if (query.split(' ').length > 3) {
            if (sentiment.score < 0) {
              const negativeResponse = nlp.generateComplexSentence("I sense", "frustration concerns", 10);
              response += " " + negativeResponse;
              console.log("Negative Sentiment Response Added:", negativeResponse);
            } else if (sentiment.score > 0) {
              const positiveResponse = nlp.generateComplexSentence("I'm glad", "positive specific discuss", 10);
              response += " " + positiveResponse;
              console.log("Positive Sentiment Response Added:", positiveResponse);
            }
          }
  
          // Identify and address relevant topics within the query
          const relevantTopics = topics.filter(topic => query.toLowerCase().includes(topic));
          relevantTopics.forEach(topic => {
            if (nlp.knowledgeBase.has(topic)) {
              const knowledgeResponse = nlp.generateComplexSentence(topic, nlp.knowledgeBase.get(topic)!, 15);
              response += " " ;
              console.log(`Knowledge Response for topic "${topic}" Added:`, knowledgeResponse);
            }
          });
  
          // If entities are recognized, append information about them to the response
          if (Object.values(recognizedEntities).some(arr => arr.length > 0)) {
            const entityDetails = Object.entries(recognizedEntities)
              .filter(([, arr]) => arr.length > 0)
              .map(([type, arr]) => `${type}(s): ${arr.join(', ')}`)
              .join('; ');
            response += ` I noticed you mentioned: ${entityDetails}`;
            console.log("Entity Information Added to Response:", entityDetails);
          }
  
          // Update the conversation history with the AI's response
          nlp.updateConversationHistory('ai', response);
          console.log("AI Response Updated in Conversation History.");
  
          return response;
        } else {
          // If no intent matches, generate a default uncertain response
          const defaultResponse = nlp.generateComplexSentence("I'm not sure I understand", query, 500);
          console.log("Default Uncertain Response Generated:", defaultResponse);
          nlp.updateConversationHistory('ai', defaultResponse);
          return defaultResponse;
        }
      }
  
  console.log("Mazs AI v1.5.0 with advanced NLP and contextual analysis capabilities initialized!");
  
  
  const intents: Intent[] = [
    {
      patterns: ['hello', 'hi', 'hey', 'hola', 'bonjour'],
      responses: [
        'Hello! How can I help you today? Check out our latest model at [MazsAI]',
        'Hi there! What can I do for you? Check out our latest model at [MazsAI]',
        'Greetings! How may I assist you? Check out our latest model at [MazsAI]'
      ],
    },
    {
      patterns: ['bye', 'goodbye', 'see you'],
      responses: ['Goodbye! Have a great day!', 'See you later! Take care!', 'Farewell! Feel free to return if you have more questions.'],
    },
    {
      patterns: ['what is gmtstudio', 'tell me about gmtstudio'],
      responses: ['GMTStudio is a platform that offers various services, including an AI WorkSpace and a social media platform called Theta.'],
    },
  
    {
      patterns: ['how are you', 'how\'s it going'],
      responses: ['I\'m doing well, thank you! How about you?', 'I\'m fine, thanks for asking! How can I help you today?'],
    },
    {
      patterns: ['tell me a joke', 'say something funny'],
      responses: ["Why don't scientists trust atoms? Because they make up everything!", 'What do you call a fake noodle? An impasta!'],
    },
    {
      patterns: ['what can you do', 'what are your capabilities'],
      responses: ['I can answer questions about GMTStudio, provide information on various topics, and even tell jokes! How can I assist you today?'],
    },
    {
      patterns: ['favorite color', 'what color do you like'],
      responses: ["As an AI, I don't have personal preferences, but I find all colors fascinating in their own way!"],
  
    },
    {
      patterns: ['Quack ', 'quack'],
      responses: ["Quack ", "Quack Quack", "I'm sorry, I can't answer that.", "Quack Quack Quack"],
    },
    {
      patterns: ['poem', 'write a poem'],
      responses: ["I'm sorry, I can't write poems."],
    },
    {
      patterns: [
        'how many {letter} are in {word}',
        'count {letter} in {word}',
        'number of {letter} in {word}',
        'how many {letters} are in {word}',
        'count {letters} in {word}',
        'number of {letters} in {word}',
        'how many {letter} are there in {word}',
      ],
      responses: ['{{countLetter}}']
  
    },
    {
      patterns: ['what is ai', 'tell me about ai'],
      responses: ['AI is a technology that allows machines to learn and make decisions based on data and algorithms.'],
    },
    {
      patterns: ['what is NLP', 'what is natural language processing'],
      responses: ['NLP is a technology that allows machines to understand and process human language.'],
    }
  ];
  
  const network = new MultilayerPerceptron([10, 32, 64, 32, intents.length], ['relu', 'relu', 'relu', 'sigmoid']);
  
  function trainNetwork() {
    const epochs = 1;  
    const learningRate = 0.1;
  
    for (let epoch = 0; epoch < epochs; epoch++) {
      let totalLoss = 0;
  
      intents.forEach((intent, intentIndex) => {
        intent.patterns.forEach(pattern => {
          const input = encodeInput(pattern);
          const target = Array(intents.length).fill(0);
          target[intentIndex] = 1;
  
          network.train(input, target, learningRate);
  
          const prediction = network.predict(input);
          totalLoss += prediction.reduce((sum, output, i) => sum + Math.pow(output - target[i], 2), 0);
        });
      });
  
      if (epoch % 100 === 0) {
        console.log(`Epoch ${epoch}, Loss: ${totalLoss}`);
      }
    }
  }
  
  function encodeInput(query: string): number[] {
    const words = query.toLowerCase().split(/\s+/);
    const wordSet = new Set(words);
    return intents.map(intent => 
      intent.patterns.some(pattern => 
        pattern.split(/\s+/).some(word => wordSet.has(word))
      ) ? 1 : 0
    );
  }
  
  const nlp = new NaturalLanguageProcessor();
  
  // Train the NLP model
  intents.forEach(intent => {
    intent.patterns.forEach(pattern => nlp.trainOnText(pattern));
    intent.responses.forEach(response => nlp.trainOnText(response));
  });
  
  // Add this near the top of the file
  let typingSpeed = 10; // Default typing speed in milliseconds
  
  // Add this function to allow changing the typing speed
  export function setTypingSpeed(speed: number) {
    typingSpeed = speed;
  }
  
  // Modify the getTypedResponse function to use the configurable typing speed
  export function getTypedResponse(response: string): Promise<string> {
    return new Promise((resolve) => {
      let typedResponse = '';
      let index = 0;
  
      function typeChar() {
        if (index < response.length) {
          typedResponse += response[index];
          index++;
          setTimeout(typeChar, typingSpeed);
        } else {
          resolve(typedResponse);
        }
      }
  
      typeChar();
    });
  }
  
  
  // Core AI functionality
  function isCodeGenerationRequest(userInput: string): boolean {
    const codePatterns = [
      /write a (.+?) code/i,
      /generate a (.+?) code/i,
      /create a (.+?) script/i,
      /can you (.+?) code/i,
      /show me how to (.+?) in code/i,
      /implement a (.+?) function/i,
      /develop a (.+?) application/i,
      /build a (.+?) module/i,
      /compose a (.+?) class/i,
      /design a (.+?) algorithm/i,
    ];
    return codePatterns.some((pattern) => pattern.test(userInput));
  }
  
  function generateCode(userInput: string): string {
    // Initialize code generation context
    const context = {
      patterns: extractPatterns(userInput),
      complexity: analyzeComplexity(userInput),
      language: detectProgrammingLanguage(userInput),
      requirements: extractRequirements(userInput)
    };
  
    // Generate optimized code based on context
    let code = '';
    
    // Add intelligent imports based on requirements
    code += generateSmartImports(context.requirements);
    
    // Generate main code structure
    if (context.patterns.includes('algorithm')) {
      code += generateOptimizedAlgorithm(context);
    } else if (context.patterns.includes('dataStructure')) {
      code += generateEfficientDataStructure(context);
    } else if (context.patterns.includes('utility')) {
      code += generateUtilityFunction(context);
    } else {
      code += generateSmartSolution(context);
    }
  
    // Add error handling and input validation
    code = addRobustErrorHandling(code, context);
  
    // Optimize the generated code
    code = optimizeCode(code);
  
    return code;
  }
  
  function extractPatterns(input: string): string[] {
    const patterns = [];
    if (input.match(/sort|search|find|algorithm/i)) patterns.push('algorithm');
    if (input.match(/array|list|tree|graph|queue|stack/i)) patterns.push('dataStructure');
    if (input.match(/helper|util|convert|format/i)) patterns.push('utility');
    return patterns;
  }
  
  function analyzeComplexity(input: string): string {
    if (input.match(/efficient|optimize|fast|quick/i)) return 'O(n)';
    if (input.match(/simple|basic/i)) return 'O(n^2)';
    return 'O(n log n)';
  }
  
  function detectProgrammingLanguage(input: string): string {
    const languageMatch = input.match(/in (javascript|typescript|python|java|c\+\+|ruby|go)/i);
    return languageMatch ? languageMatch[1].toLowerCase() : 'typescript';
  }
  
  function extractRequirements(input: string): string[] {
    const requirements: string[] = [];
    if (input.match(/math|calculate|compute/i)) requirements.push('math');
    if (input.match(/data|process|transform/i)) requirements.push('data');
    if (input.match(/async|promise|await/i)) requirements.push('async');
    return requirements;
  }
  
  function generateSmartImports(requirements: string[]): string {
    let imports = '';
    if (requirements.includes('math')) imports += 'import { Math } from "./math";\n';
    if (requirements.includes('data')) imports += 'import { DataProcessor } from "./data";\n';
    if (requirements.includes('async')) imports += 'import { AsyncHandler } from "./async";\n';
    return imports;
  }
  
  function generateOptimizedAlgorithm(context: any): string {
    const { complexity, requirements } = context;
    let code = '';
    
    if (complexity === 'O(n)') {
      code += generateLinearTimeAlgorithm(requirements);
    } else if (complexity === 'O(n log n)') {
      code += generateLogLinearTimeAlgorithm(requirements);
    }
    
    return code;
  }
  
  function generateLinearTimeAlgorithm(requirements: string[]): string {
    return `function processLinear(data: any[]): any[] {
    const result = new Map();
    for (const item of data) {
      result.set(item, process(item));
    }
    return Array.from(result.values());
  }`;
  }
  
  function generateLogLinearTimeAlgorithm(requirements: string[]): string {
    return `function processLogLinear(data: any[]): any[] {
    return data
      .sort((a, b) => compare(a, b))
      .map(item => process(item));
  }`;
  }
  
  function generateEfficientDataStructure(context: any): string {
    const { requirements } = context;
    let code = '';
    
    if (requirements.includes('search')) {
      code += generateSearchStructure();
    } else if (requirements.includes('storage')) {
      code += generateStorageStructure();
    }
    
    return code;
  }
  
  function generateSearchStructure(): string {
    return `class SearchOptimizedStructure<T> {
    private data: Map<string, T> = new Map();
    
    add(key: string, value: T): void {
      this.data.set(key, value);
    }
    
    find(key: string): T | undefined {
      return this.data.get(key);
    }
  }`;
  }
  
  function generateStorageStructure(): string {
    return `class StorageOptimizedStructure<T> {
    private data: T[] = [];
    private index: Map<string, number> = new Map();
    
    store(key: string, value: T): void {
      const pos = this.data.length;
      this.data.push(value);
      this.index.set(key, pos);
    }
    
    retrieve(key: string): T | undefined {
      const pos = this.index.get(key);
      return pos !== undefined ? this.data[pos] : undefined;
    }
  }`;
  }
  
  function generateUtilityFunction(context: any): string {
    return `function processData(input: any): any {
    if (!isValid(input)) throw new Error('Invalid input');
    
    switch(typeof input) {
      case 'string': return optimizeString(input);
      case 'number': return optimizeNumber(input);
      case 'object': return optimizeObject(input);
      default: return input;
    }
  }`;
  }
  
  function generateSmartSolution(context: any): string {
    return `class SmartProcessor {
    private cache = new Map();
    
    process(input: any): any {
      if (this.cache.has(input)) return this.cache.get(input);
      const result = this.computeOptimizedResult(input);
      this.cache.set(input, result);
      return result;
    }
    
    private computeOptimizedResult(input: any): any {
      return input;
    }
  }`;
  }
  
  function addRobustErrorHandling(code: string, context: any): string {
    return `try {
    if (!input) throw new Error('Invalid input');
    ${code}
  } catch (error) {
    console.error('Error:', error);
    throw error;
  }`;
  }
  
  function optimizeCode(code: string): string {
    return code.split('\n').join('\n');
  }
  
  interface CodeComponent {
    type: 'function' | 'class' | 'variable' | 'loop' | 'condition' | 'import';
    name?: string;
    params?: string[];
    returnType?: string;
    body?: string[];
    condition?: string;
  }
  
  function analyzeRequirements(requirements: string): CodeComponent[] {
    const components: CodeComponent[] = [];
    
    // Detect required imports
    if (requirements.match(/math|calculation|compute|calculate/i)) {
      components.push({ type: 'import', name: 'math' });
    }
    
    // Detect loops
    if (requirements.match(/each|every|iterate|loop|repeat/i)) {
      components.push({
        type: 'loop',
        condition: 'i < items.length',
        body: ['// Process each item']
      });
    }
  
    // Detect conditions
    if (requirements.match(/if|when|check|verify|validate/i)) {
      components.push({
        type: 'condition',
        condition: '// Add condition here',
        body: ['// Handle condition']
      });
    }
  
    // Detect data processing
    if (requirements.match(/sort|filter|map|reduce|transform/i)) {
      components.push({
        type: 'function',
        name: 'processData',
        params: ['data'],
        returnType: 'array',
        body: ['// Process data here']
      });
    }
  
    return components;
  }
  
  function generateCodeFromComponents(language: string, functionality: string, components: CodeComponent[]): string {
    const code: string[] = [];
    const indent = language === 'python' ? '    ' : '  ';
    
    // Add imports
    components.filter(c => c.type === 'import').forEach(imp => {
      if (language === 'python') {
        code.push(`import ${imp.name}`);
      } else {
        code.push(`const ${imp.name} = require('${imp.name}');`);
      }
    });
    
    // Add main function/class
    const mainName = language === 'python' ? snake_case(functionality) : camelCase(functionality);
    if (language === 'python') {
      code.push('');
      code.push(`def ${mainName}():`);
      code.push(`${indent}"""`);
      code.push(`${indent}${functionality.charAt(0).toUpperCase() + functionality.slice(1)}`);
      code.push(`${indent}"""`);
    } else {
      code.push('');
      code.push(`function ${mainName}() {`);
      code.push(`${indent}// ${functionality}`);
    }
  
    // Add component implementations
    components.forEach(component => {
      switch (component.type) {
        case 'loop':
          if (language === 'python') {
            code.push(`${indent}for item in items:`);
            component.body?.forEach(line => code.push(`${indent}${indent}${line}`));
          } else {
            code.push(`${indent}for (const item of items) {`);
            component.body?.forEach(line => code.push(`${indent}${indent}${line}`));
            code.push(`${indent}}`);
          }
          break;
  
        case 'condition':
          if (language === 'python') {
            code.push(`${indent}if ${component.condition}:`);
            component.body?.forEach(line => code.push(`${indent}${indent}${line}`));
          } else {
            code.push(`${indent}if (${component.condition}) {`);
            component.body?.forEach(line => code.push(`${indent}${indent}${line}`));
            code.push(`${indent}}`);
          }
          break;
  
        case 'function':
          if (component.name !== mainName) {
            code.push('');
            if (language === 'python') {
              code.push(`${indent}def ${component.name}(${component.params?.join(', ') || ''}):`);
              component.body?.forEach(line => code.push(`${indent}${indent}${line}`));
            } else {
              code.push(`${indent}function ${component.name}(${component.params?.join(', ') || ''}) {`);
              component.body?.forEach(line => code.push(`${indent}${indent}${line}`));
              code.push(`${indent}}`);
            }
          }
          break;
      }
    });
  
    // Add return statement
    if (language === 'python') {
      code.push(`${indent}return result`);
    } else {
      code.push(`${indent}return result;`);
      code.push('}');
    }
  
    // Add example usage
    code.push('');
    code.push('// Example usage:');
    if (language === 'python') {
      code.push(`if __name__ == "__main__":`);
      code.push(`${indent}result = ${mainName}()`);
      code.push(`${indent}print(f"Result: {result}")`);
    } else {
      code.push(`const result = ${mainName}();`);
      code.push('console.log("Result:", result);');
    }
  
    return formatCode(code.join('\n'), language);
  }
  
  // Helper functions for different naming conventions
  function camelCase(str: string): string {
    return str.replace(/(?:^\w|[A-Z]|\b\w)/g, (word, index) => 
      index === 0 ? word.toLowerCase() : word.toUpperCase()
    ).replace(/\s+/g, '');
  }
  
  
  function snake_case(str: string): string {
    return str.toLowerCase().replace(/\s+/g, '_');
  }
    
  /**
   * Capitalizes the first letter of a given word.
   * 
   * @param word - The word to capitalize.
   * @returns The capitalized word.
   */
  
  /**
   * Formats the generated code based on the programming language.
   * This ensures proper indentation and syntax.
   * 
   * @param code - The raw generated code.
   * @param language - The programming language of the code.
   * @returns The formatted code.
   */
  function formatCode(code: string, language: string): string {
    const lines = code.split('\n');
    const formattedLines: string[] = [];
    let indentLevel = 0;
    const indentSize = 2;
    
    lines.forEach(line => {
      line = line.trim();
      if (line.match(/^\}/) || line.match(/^end/)) {
        indentLevel = Math.max(indentLevel - 1, 0);
      }
      
      const indent = ' '.repeat(indentLevel * indentSize);
      formattedLines.push(indent + line);
      
      if (line.match(/\{$/) || (language === 'python' && line.match(/:$/))) {
        indentLevel++;
      }
    });
    
    return formattedLines.join('\n');
  }
  // Modify the Input function to use the configurable typing speed
  // Modify the handleUserInput function to include puzzle-solving
  export function handleUserInput(userInput: string, targetLanguage?: string): Promise<string> {
    console.log("User:", userInput);
    nlp.updateContext(userInput);
    
    return new Promise((resolve) => {
      setTimeout(() => {
        let response: string;
        let responseType: 'text' | 'code' = 'text';
        
        if (isCodeGenerationRequest(userInput)) {
          // Generate code
          response = generateCode(userInput);
          responseType = 'code';
        } else if (/please summarize this/i.test(userInput)) {
          // Enhanced text summarization logic
          const textToSummarize = userInput.replace(/please summarize this/i, '').trim();
          response = enhancedSummarizeText(textToSummarize);
          
        } else if(/please continue this sentence/i.test(userInput)) {
          const partialSentence = userInput.replace(/please continue this sentence/i, '').trim();
          response = continueSentence(partialSentence);
        } else if (/how many\s+[a-zA-Z]\s+(are|is)?\s*(in|within)?\s*(the\s+word\s+)?\w+/i.test(userInput)) {
          response = nlp.countLetterInWord(userInput);
        } else if (userInput.toLowerCase().startsWith("no !")) {
          response = generateApologyResponse(userInput.slice(4).trim());
        } else if (/[+\-*/%^]/.test(userInput)) {
          response = generateComplexMathCalculation(userInput);
        } else if (isPuzzle(userInput)) {
          response = solvePuzzle(userInput);
        } else if (isMorseCode(userInput)) {
          response = processMorseCode(userInput);
        } else if (/finish this sentence\s*:/i.test(userInput)) {
          const partialSentence = userInput.split(/:\s*/)[1].trim();
          response = continueSentence(partialSentence);
        } else {
          response = processChatbotQuery(userInput);
        }
        
        if (targetLanguage) {
          response = nlp.translateText(response, targetLanguage);
        }
        
        // Attach the response type
        resolve(JSON.stringify({ text: response, type: responseType }));
      }, 100);
    });
  }
  
  function continueSentence(partialSentence: string): string {
    if (!partialSentence || partialSentence.trim().length === 0) {
      return "Please provide a partial sentence for me to continue.";
    }
    // Enhanced n-gram model for improved sentence continuation
    class EnhancedNGramModel {
      private nGrams: Map<string, string[]>;
      private n: number;
      private vocabulary: Set<string>;
      public sentenceEndings: Set<string>;
      public punctuation: string[];
  
      constructor(n: number = 4) {
        this.nGrams = new Map();
        this.n = n;
        this.vocabulary = new Set();
        this.sentenceEndings = new Set(['.', '!', '?']);
        this.punctuation = [',', ';', ':', '-', '(', ')', '"', "'"];
      }
  
      train(text: string): void {
        const sentences = text.match(/[^.!?]+[.!?]+/g) || [text];
        sentences.forEach(sentence => {
          const words = sentence.toLowerCase().split(/\s+/);
          words.forEach(word => {
            const cleanWord = word.replace(/[^a-zA-Z]/g, '');
            if (cleanWord) {
              this.vocabulary.add(cleanWord);
            }
          });
          for (let i = 0; i <= words.length - this.n; i++) {
            const context = words.slice(i, i + this.n - 1).join(' ');
            const nextWord = words[i + this.n - 1];
            if (!this.nGrams.has(context)) {
              this.nGrams.set(context, []);
            }
            this.nGrams.get(context)!.push(nextWord);
          }
        });
      }
  
      predict(context: string): string {
        const contextWords = context.toLowerCase().split(/\s+/).slice(-this.n + 1).join(' ');
        if (this.nGrams.has(contextWords)) {
          const possibleNextWords = this.nGrams.get(contextWords)!;
          const wordFrequency: { [key: string]: number } = {};
          possibleNextWords.forEach(word => {
            wordFrequency[word] = (wordFrequency[word] || 0) + 1;
          });
          const words = Object.keys(wordFrequency);
          const frequencies = Object.values(wordFrequency);
          const total = frequencies.reduce((a, b) => a + b, 0);
          let rand = Math.random() * total;
          for (let i = 0; i < words.length; i++) {
            if (rand < frequencies[i]) {
              return words[i];
            }
            rand -= frequencies[i];
          }
        }
        // Fallback to using a larger context or random word
        for (let currentN = this.n - 2; currentN >= 1; currentN--) {
          const fallbackContext = context.toLowerCase().split(/\s+/).slice(-currentN).join(' ');
          if (this.nGrams.has(fallbackContext)) {
            const possibleNextWords = this.nGrams.get(fallbackContext)!;
            return possibleNextWords[Math.floor(Math.random() * possibleNextWords.length)];
          }
        }
        // Final fallback to a random word from the vocabulary
        const vocabArray = Array.from(this.vocabulary);
        return vocabArray[Math.floor(Math.random() * vocabArray.length)];
      }
    }
  
    // Initialize and train the enhanced n-gram model with expanded training text
    const trainingText = `
      Artificial intelligence is revolutionizing various industries and transforming the way we live and work.
      Machine learning algorithms can analyze vast amounts of data to uncover patterns and make predictions.
      Natural language processing enables computers to understand, interpret, and generate human language.
      Deep learning models have achieved remarkable results in image recognition and speech synthesis tasks.
      The ethical implications of AI development are a subject of ongoing debate among researchers and policymakers.
      Reinforcement learning allows AI agents to learn optimal strategies through trial and error in complex environments.
      Computer vision systems can now detect objects, recognize faces, and interpret visual scenes with high accuracy.
      The integration of AI in healthcare has the potential to improve diagnosis, treatment, and patient care.
      Autonomous vehicles powered by AI technologies are poised to transform transportation and urban planning.
      Explainable AI aims to make machine learning models more transparent and interpretable for human users.
      Generative AI models can create realistic images, videos, and text based on learned patterns and user prompts.
      Federated learning enables AI models to be trained across multiple decentralized devices while preserving privacy.
      Quantum computing has the potential to significantly accelerate certain AI algorithms and computations.
      Edge AI brings machine learning capabilities directly to IoT devices, reducing latency and improving efficiency.
      AI-powered robotics is advancing rapidly, with applications in manufacturing, healthcare, and space exploration.
      Advanced algorithms facilitate the seamless integration of AI systems into existing infrastructures.
      Ethical AI ensures that technologies are developed and implemented responsibly and without bias.
      Collaborative efforts between humans and AI can lead to groundbreaking innovations and solutions.
      The scalability of AI models is crucial for their deployment in real-world applications.
      Continuous learning allows AI systems to adapt to new information and evolving environments.
      Hello, I am a human. This is a test for an AI to respond to user inputs effectively and accurately.
      The global economy is interconnected, with trade relationships spanning continents and oceans.
      Climate change poses significant challenges to ecosystems and human societies worldwide.
      Renewable energy sources are becoming increasingly important in the transition to sustainable power.
      Biodiversity loss threatens the delicate balance of ecosystems and food chains.
      Urban planning focuses on creating sustainable, livable cities for growing populations.
      Cultural exchange enriches societies and promotes understanding between diverse groups.
      Education systems are evolving to meet the needs of a rapidly changing job market.
      Public health initiatives aim to improve well-being and prevent the spread of diseases.
      Technological advancements in agriculture are helping to address food security concerns.
      Space exploration continues to push the boundaries of human knowledge and capabilities.
      Ocean conservation efforts are crucial for maintaining marine biodiversity and resources.
      The arts play a vital role in expressing human experiences and emotions across cultures.
      Economic policies shape the distribution of wealth and opportunities within societies.
      Sustainable transportation solutions are being developed to reduce carbon emissions.
      Historical preservation helps maintain connections to cultural heritage and past events.
      Diplomatic relations between nations influence global stability and cooperation.
      Advancements in medicine are improving life expectancy and quality of life worldwide.
      The fashion industry is grappling with issues of sustainability and ethical production.
      Sports bring people together and promote physical fitness and friendly competition.
      Linguistic diversity is celebrated as a valuable aspect of human cultural heritage.
      Innovation drives progress, enabling societies to overcome challenges and achieve new heights.
      Technology integration is essential for optimizing productivity and fostering economic growth.
      Sustainable practices contribute to environmental conservation and long-term viability.
      Human creativity and technological advancements go hand in hand to shape the future.
      Data-driven decisions enhance efficiency and effectiveness across various sectors.
      The synergy between humans and machines can lead to unparalleled achievements.
    `;
    const model = new EnhancedNGramModel(4);
    model.train(trainingText);
  
    // Generate sentence continuation one word at a time
    const tokens = partialSentence.toLowerCase().split(/\s+/);
    let continuation = '';
    const maxWords = 25; // Increased word limit for more detailed sentences
  
    for (let i = 0; i < maxWords; i++) {
      const context = tokens.slice(-3).join(' ');
      const nextWord = model.predict(context);
  
      if (model.sentenceEndings.has(nextWord)) {
        continuation += nextWord;
        break;
      } else if (model.punctuation.includes(nextWord)) {
        continuation = continuation.trim() + nextWord + ' ';
      } else {
        continuation += ' ' + nextWord;
        tokens.push(nextWord);
      }
    }
  
    // Post-process the continuation
    continuation = continuation.trim();
    continuation = continuation.charAt(0).toUpperCase() + continuation.slice(1);
  
    // Ensure proper sentence ending
    if (!/[.!?]$/.test(continuation)) {
      continuation += '.';
    }
  
    return `${partialSentence} ${continuation}`;
  }
  function enhancedSummarizeText(text: string): string {
    // Predefined stopwords list
    const stopwords = new Set([
      'the', 'and', 'is', 'in', 'at', 'of', 'a', 'an', 'on', 'for', 'to', 'with', 'by',
      'this', 'that', 'from', 'or', 'as', 'it', 'be', 'are', 'was', 'were', 'has',
      'have', 'had', 'but', 'not', 'they', 'their', 'can', 'will', 'would', 'there',
      'what', 'which', 'who', 'whom', 'when', 'where', 'why', 'how'
    ]);
  
    // Function to preprocess text and calculate word frequencies
    function preprocess(text: string): { sentences: string[]; wordFrequencies: { [key: string]: number } } {
      // Split the text into sentences using punctuation as delimiters
      const sentenceRegex = /[^.!?]+[.!?]+/g;
      const sentences = text.match(sentenceRegex) || [text.trim()];
  
      // Initialize word frequencies
      const wordFrequencies: { [key: string]: number } = {};
  
      // Process each sentence
      sentences.forEach(sentence => {
        // Remove punctuation, convert to lowercase, and split into words
        const words = sentence.toLowerCase().replace(/[^a-zA-Z\s]/g, '').split(/\s+/);
  
        words.forEach(word => {
          if (word.length === 0 || stopwords.has(word)) return;
          wordFrequencies[word] = (wordFrequencies[word] || 0) + 1;
        });
      });
  
      return { sentences, wordFrequencies };
    }
  
    // Function to score each sentence based on word frequencies
    function scoreSentences(
      sentences: string[],
      wordFrequencies: { [key: string]: number }
    ): { sentence: string; score: number; index: number }[] {
      return sentences.map((sentence, index) => {
        const words = sentence.toLowerCase().replace(/[^a-zA-Z\s]/g, '').split(/\s+/);
        let score = 0;
  
        words.forEach(word => {
          if (word.length === 0 || stopwords.has(word)) return;
          if (wordFrequencies[word]) {
            score += wordFrequencies[word];
          }
        });
  
        // Additional scoring factors
        // Give higher scores to sentences at the beginning of the text
        const positionFactor = (sentences.length - index) / sentences.length;
        score += positionFactor * 2; // Weight the position factor
  
        // Optionally, prefer longer sentences as they may contain more information
        const lengthFactor = words.length / 20; // Normalize based on average sentence length
        score += lengthFactor;
  
        return { sentence: sentence.trim(), score, index };
      });
    }
  
    // Function to generate the summary from scored sentences
    function summarize(
      sentences: string[],
      scoredSentences: { sentence: string; score: number; index: number }[]
    ): string {
      const summaryPercentage = 0.3;
      const summaryLength = Math.max(1, Math.floor(sentences.length * summaryPercentage));
  
      // Sort sentences by score in descending order without mutating the original array
      const sortedScoredSentences = [...scoredSentences].sort((a, b) => b.score - a.score);
  
      // Select top scoring sentences
      const topScoredSentences = sortedScoredSentences.slice(0, summaryLength);
  
      // Sort the selected sentences back to their original order
      const orderedTopSentences = topScoredSentences.sort((a, b) => a.index - b.index);
  
      // Combine the selected sentences into a summary
      const summary = orderedTopSentences.map(item => item.sentence).join(' ');
  
      return summary;
    }
  
    // Main Summarization Process
    const { sentences, wordFrequencies } = preprocess(text);
    const scoredSentences = scoreSentences(sentences, wordFrequencies);
    const summary = summarize(sentences, scoredSentences);
  
    // Handle cases where the summary might be too short or empty
    const finalSummary = summary.length > 0 ? summary : "Unable to generate a meaningful summary from the provided text.";
  
    return `Summary:
  
  ${finalSummary}
  
  Would you like me to elaborate on any part of the summary?`;
  }
  
  
  // Function to check if the input is Morse code
  function isMorseCode(input: string): boolean {
    // Morse code consists of dots, dashes, and spaces
    return /^[.\- ]+$/.test(input.trim());
  }
  
  // Function to process Morse code
  function processMorseCode(input: string): string {
    const morseToText = decodeMorseCode(input);
    const textToMorse = encodeMorseCode(input);
    
    if (morseToText !== input) {
      return `Decoded Morse code: "${morseToText}"
  
  Here's an explanation of the decoding process:
  1. Each Morse code character is separated by a space.
  2. Words are separated by three spaces.
  3. The decoder matches each Morse code character to its corresponding letter or number.
  4. The decoded characters are then combined to form the final message.
  
  Would you like to encode a message into Morse code?`;
    } else if (textToMorse !== input) {
      return `Encoded Morse code: "${textToMorse}"
  
  Here's an explanation of the encoding process:
  1. Each letter and number in the input is converted to its Morse code equivalent.
  2. Morse code characters are separated by a single space.
  3. Words are separated by three spaces.
  4. The encoded characters are then combined to form the final Morse code message.
  
  Would you like to decode a Morse code message?`;
    } else {
      return "I'm sorry, but I couldn't recognize this as valid Morse code or text to encode. Please make sure you're using only dots (.) and dashes (-) for Morse code, or plain text for encoding.";
    }
  }
  
  // Function to decode Morse code to text
  function decodeMorseCode(morseCode: string): string {
    const morseCodeMap: { [key: string]: string } = {
      '.-': 'A', '-...': 'B', '-.-.': 'C', '-..': 'D', '.': 'E', '..-.': 'F',
      '--.': 'G', '....': 'H', '..': 'I', '.---': 'J', '-.-': 'K', '.-..': 'L',
      '--': 'M', '-.': 'N', '---': 'O', '.--.': 'P', '--.-': 'Q', '.-.': 'R',
      '...': 'S', '-': 'T', '..-': 'U', '...-': 'V', '.--': 'W', '-..-': 'X',
      '-.--': 'Y', '--..': 'Z',
      '-----': '0', '.----': '1', '..---': '2', '...--': '3', '....-': '4',
      '.....': '5', '-....': '6', '--...': '7', '---..': '8', '----.': '9',
      '.-.-.-': '.', '--..--': ',', '..--..': '?', '.----.': "'", '-.-.--': '!',
      '-..-.': '/', '-.--.': '(', '-.--.-': ')', '.-...': '&', '---...': ':',
      '-.-.-.': ';', '-...-': '=', '.-.-.': '+', '-....-': '-', '..--.-': '_',
      '.-..-.': '"', '...-..-': '$', '.--.-.': '@', '...---...': 'SOS'
    };
  
    return morseCode
      .split('   ')
      .map(word => word
        .split(' ')
        .map(char => morseCodeMap[char] || '')
        .join('')
      )
      .join(' ');
  }
  
  // Function to encode text to Morse code
  function encodeMorseCode(text: string): string {
    const textToMorseMap: { [key: string]: string } = {
      'A': '.-', 'B': '-...', 'C': '-.-.', 'D': '-..', 'E': '.', 'F': '..-.',
      'G': '--.', 'H': '....', 'I': '..', 'J': '.---', 'K': '-.-', 'L': '.-..',
      'M': '--', 'N': '-.', 'O': '---', 'P': '.--.', 'Q': '--.-', 'R': '.-.',
      'S': '...', 'T': '-', 'U': '..-', 'V': '...-', 'W': '.--', 'X': '-..-',
      'Y': '-.--', 'Z': '--..', 
      '0': '-----', '1': '.----', '2': '..---', '3': '...--', '4': '....-',
      '5': '.....', '6': '-....', '7': '--...', '8': '---..', '9': '----.',
      '.': '.-.-.-', ',': '--..--', '?': '..--..', "'": '.----.', '!': '-.-.--',
      '/': '-..-.', '(': '-.--.', ')': '-.--.-', '&': '.-...', ':': '---...',
      ';': '-.-.-.', '=': '-...-', '+': '.-.-.', '-': '-....-', '_': '..--.-',
      '"': '.-..-.', '$': '...-..-', '@': '.--.-.', 'SOS': '...---...'
    };
  
    return text
      .toUpperCase()
      .split(' ')
      .map(word => word
        .split('')
        .map(char => textToMorseMap[char] || '')
        .join(' ')
      )
      .join('   ');
  }
  // Add this function to check if the input is a puzzle
  function isPuzzle(input: string): boolean {
    const puzzleKeywords = [
      'puzzle',
      'riddle',
      'what has',
      'what comes',
      'if you have',
      // Add more puzzle-related keywords or patterns here
    ];
    return puzzleKeywords.some(keyword => input.toLowerCase().includes(keyword));
  }
  function generateApologyResponse(userMessage: string): string {
    const apologies = [
      "I apologize for any confusion. ",
      "I'm sorry if I misunderstood. ",
      "My apologies for the mistake. ",
      "I regret any inconvenience caused. ",
    ];
  
    const followUps = [
      "Could you please provide more details about what you meant?",
      "How can I better assist you with your request?",
      "Would you mind rephrasing your question? I want to make sure I understand correctly.",
      "I'd like to help you better. Can you elaborate on your needs?",
    ];
  
    const apology = apologies[Math.floor(Math.random() * apologies.length)];
    const followUp = followUps[Math.floor(Math.random() * followUps.length)];
  
    return `${apology}${followUp}`;
  }
  
  function generateComplexMathCalculation(input: string): string {
    try {
      // Sanitize the input to remove any potential harmful code
      const sanitizedInput = input.replace(/[^0-9+\-*/%^().\s]/g, '');
      
      // Evaluate the expression
      const result = eval(sanitizedInput);
      
      // Generate a more complex response
      const complexResponse = `The Answer is ${result}.
  
  here is the calculation process, first, Let's break down this calculation step by step:
  
  Input: ${sanitizedInput}
  
  Step 1: Identify the operations
  
  We have the following operations: ${identifyOperations(sanitizedInput)}
  
  Step 2: Apply order of operations (PEMDAS)
  
  1. Parentheses
  2. Exponents
  3. Multiplication and Division (left to right)
  4. Addition and Subtraction (left to right)
  
  Step 3: Evaluate
  
  ${generateSteps(sanitizedInput)}
  
  Final Result: ${result}
  
  Is there anything else you'd like to calculate?`;
      
      return complexResponse.trim();
    } catch (error) {
      return "I apologize, but I couldn't process that mathematical expression. Please make sure it's a valid arithmetic operation and try again.";
    }
  }
  
  function identifyOperations(input: string): string {
    const operations = [];
    if (input.includes('+')) operations.push('Addition');
    if (input.includes('-')) operations.push('Subtraction');
    if (input.includes('*')) operations.push('Multiplication');
    if (input.includes('/')) operations.push('Division');
    if (input.includes('%')) operations.push('Modulus');
    if (input.includes('^')) operations.push('Exponentiation');
    
  
    return operations.join(', ');
  }
  
  function generateSteps(input: string): string {
    // This is a simplified version. In a real-world scenario, you'd want to implement
    // a proper expression parser and evaluator to show detailed steps.
    const steps = [];
    let currentStep = input;
    while (currentStep.includes('(')) {
      const match = currentStep.match(/\(([^()]+)\)/);
      if (match) {
        const subExpr = match[1];
        const subResult = eval(subExpr);
        steps.push(`Evaluate (${subExpr}) = ${subResult}`);
        currentStep = currentStep.replace(`(${subExpr})`, subResult.toString());
      }
    }
    steps.push(`Evaluate remaining expression: ${currentStep} = ${eval(currentStep)}`);
    return steps.join('\n');
  }
  
  
  // Add the regenerateResponse function
  export function regenerateResponse(userInput: string): Promise<string> {
    console.log("Regenerating response for:", userInput);
    nlp.updateContext(userInput);
    return new Promise((resolve) => {
      setTimeout(() => {
        const response = processChatbotQuery(userInput);
        getTypedResponse(response).then(resolve);
      }, 100); // Simulate a delay in processing
    });
  }
  
  export function getConversationSuggestions(): string[] {
    return [
      "what is 2 ** 10",
      "please continue this sentence : ",
      "please summarize this sentence : ",
      "write a python code"
      
    ];
  }
  
  export function debounce<F extends (...args: any[]) => any>(func: F, waitFor: number) {
    let timeout: ReturnType<typeof setTimeout> | null = null;
  
    return (...args: Parameters<F>): Promise<ReturnType<F>> => {
      return new Promise((resolve) => {
        if (timeout) {
          clearTimeout(timeout);
        }
  
        timeout = setTimeout(() => resolve(func(...args)), waitFor);
      });
    };
  }
  
  export const debouncedHandleUserInput = debounce(handleUserInput, 300);
  
  // Train the network when the module is loaded
  trainNetwork();
  
  console.log("Mazs AI v1.5.0 anatra");
  
  export async function processAttachedFile(file: File): Promise<string> {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
  
      reader.onload = async (event) => {
        try {
          let response = '';
          switch (file.type) {
            case 'text/plain':
              response = await processTextFile(event.target?.result as string);
              break;
            
            case 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet':
            case 'application/vnd.ms-excel':
              response = await processExcelFile(event.target?.result as ArrayBuffer);
              break;
            case 'application/json':
              response = await processJsonFile(event.target?.result as string);
              break;
            case 'application/pdf':
              response = await processPDFFile(event.target?.result as string);
              break;
            case 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
            case 'application/msword':
              response = await processDocxFile(event.target?.result as string);
              break;
            case 'image/png':
            case 'image/jpeg':
            case 'image/jpg':
            case 'image/gif':
            case 'image/heic':
            case 'image/webp':
            case 'image/svg+xml':
            case 'image/bmp':
            case 'image/tiff':
            case 'image/x-icon':
            case 'image/HEIC':
              response = await processImageFile(event.target?.result as ArrayBuffer);
              break;
            case 'audio/wav':
            case 'audio/mpeg':
            case 'audio/ogg':
            case 'audio/webm':
            case 'audio/aac':
            case 'audio/flac':
            case 'audio/aiff':
            case 'audio/m4a':
              response = await processVoiceFile(event.target?.result as ArrayBuffer);
              break;
            case 'video/mp4':
            case 'video/quicktime':
            case 'video/avi':
            case 'video/mkv':
            case 'video/webm':
            case 'video/flv':
            case 'video/wmv':
            case 'video/3gpp':
            case 'video/3gpp2':
            case 'video/mp4; codecs=hevc,mp4a.40.2':
            case 'video/hevc':
            case 'video/x-m4v':
              try {
                response = await processMediaFile(new File([event.target?.result as ArrayBuffer], file.name, { type: file.type }));
              } catch (error) {
                console.error('Error processing video file:', error);
                response = "I'm sorry, I couldn't process this video file. There was an error during processing.";
              }
              break;
            case 'application/code':
              // Handle code files if necessary
              response = await processCodeFile(event.target?.result as string);
              break;
            default:
              response = "I'm sorry, I can't process this file type.";
          }
          resolve(JSON.stringify({ text: response, type: 'text' }));
        } catch (error) {
          reject(error);
        }
      };
      reader.onerror = () => {
        reject(new Error('Error reading file'));
      };
  
      if (file.type === 'text/plain' || file.type === 'application/code') {
        reader.readAsText(file);
      } else {
        reader.readAsArrayBuffer(file);
      }
    });
  }
  
  // Example function to process code files
  async function processCodeFile(content: string): Promise<string> {
    // Implement any specific processing for code files if needed
    return `Received a code file. Here is the content:\n\n${content}`;
  }
  
  async function processTextFile(content: string): Promise<string> {
    const words = content.split(/\s+/).length;
    const lines = content.split('\n').length;
    const characters = content.length;
    const sentences = content.split(/[.!?]+/).filter(s => s.trim().length > 0);
    const sentenceCount = sentences.length;
  
    // Calculate average word length
    const avgWordLength = characters / words;
  
    // Find the most common words (excluding stop words)
    const wordFrequency = new Map<string, number>();
    const stopWords = new Set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should', 'could', 'can']);
    content.toLowerCase().split(/\s+/).forEach(word => {
      if (!stopWords.has(word)) {
        wordFrequency.set(word, (wordFrequency.get(word) || 0) + 1);
      }
    });
    const commonWords = Array.from(wordFrequency.entries())
      .sort((a, b) => b[1] - a[1])
      .slice(0, 10)
      .map(entry => entry[0]);
  
    // Estimate reading time (assuming average reading speed of 200 words per minute)
    const readingTimeMinutes = Math.ceil(words / 200);
  
    // Basic text summarization
    function summarizeText(text: string, numSentences: number = 3): string {
      const sentences = text.split(/[.!?]+/).filter(s => s.trim().length > 0);
      const wordFrequency = new Map<string, number>();
      
      sentences.forEach(sentence => {
        sentence.toLowerCase().split(/\s+/).forEach(word => {
          if (!stopWords.has(word)) {
            wordFrequency.set(word, (wordFrequency.get(word) || 0) + 1);
          }
        });
      });
  
      const sentenceScores = sentences.map(sentence => {
        const words = sentence.toLowerCase().split(/\s+/);
        const score = words.reduce((sum, word) => sum + (wordFrequency.get(word) || 0), 0) / words.length;
        return { sentence, score };
      });
  
      const topSentences = sentenceScores
        .sort((a, b) => b.score - a.score)
        .slice(0, numSentences)
        .sort((a, b) => sentences.indexOf(a.sentence) - sentences.indexOf(b.sentence))
        .map(item => item.sentence);
  
      return topSentences.join(' ');
    }
  
    const summary = summarizeText(content, 3);
  
    // Calculate sentiment
    function calculateSentiment(text: string): string {
      const positiveWords = new Set(['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'happy', 'joy', 'love', 'like', 'best']);
      const negativeWords = new Set(['bad', 'terrible', 'awful', 'horrible', 'disappointing', 'sad', 'angry', 'hate', 'dislike', 'worst']);
      
      let positiveCount = 0;
      let negativeCount = 0;
      
      text.toLowerCase().split(/\s+/).forEach(word => {
        if (positiveWords.has(word)) positiveCount++;
        if (negativeWords.has(word)) negativeCount++;
      });
      
      if (positiveCount > negativeCount) return "positive";
      if (negativeCount > positiveCount) return "negative";
      return "neutral";
    }
  
    const sentiment = calculateSentiment(content);
  
    // Generate a summary using the NLP model
    const nlpSummary = nlp.generateComplexSentence(
      "The text file analysis reveals",
      `${words} words, ${sentenceCount} sentences, ${lines} lines, ${characters} characters, common words, ${sentiment} sentiment`,
      50
    );
  
    return `I've analyzed the text file. Here's what I found:
  
  1. Word count: ${words}
  2. Sentence count: ${sentenceCount}
  3. Line count: ${lines}
  4. Character count: ${characters}
  5. Average word length: ${avgWordLength.toFixed(2)} characters
  6. Most common words: ${commonWords.join(', ')}
  7. Estimated reading time: ${readingTimeMinutes} minute${readingTimeMinutes > 1 ? 's' : ''}
  8. Overall sentiment: ${sentiment}
  
  Summary:
  ${summary}
  
  ${nlpSummary}
  
  Would you like me to perform any specific analysis on this text?`;
  }
  
  async function processPDFFile(content: string): Promise<string> {
    const text = extractTextFromPDF(content);
    return processTextFile(text);
  }
  
  async function processDocxFile(content: string): Promise<string> {
    const text = extractTextFromDocx(content);
    return processTextFile(text);
  }
  
  async function processExcelFile(content: ArrayBuffer): Promise<string> {
    // Enhanced function to process Excel file without external libraries
    try {
      // Convert ArrayBuffer to string using TextDecoder for better performance
      const decoder = new TextDecoder('utf-8');
      const csvString = decoder.decode(content);
  
      // Split the CSV into rows
      const rows = csvString.trim().split(/\r?\n/);
      const rowCount = rows.length;
  
      if (rowCount === 0) {
        return "The Excel file is empty. Please provide a file with data.";
      }
  
      // Split the first row to determine the number of columns
      const firstRow = rows[0];
      const columns = parseCSVLine(firstRow);
      const columnCount = columns.length;
  
      // Initialize data structures for analysis
      let totalCells = 0;
      let totalLength = 0;
      const wordFrequency: { [key: string]: number } = {};
      const columnData: { [key: string]: string[] } = {};
  
      // Process each row
      rows.forEach((row, rowIndex) => {
        const cells = parseCSVLine(row);
        totalCells += cells.length;
  
        cells.forEach((cell, cellIndex) => {
          const cleanedCell = cell.trim();
          totalLength += cleanedCell.length;
  
          // Split cell into words for frequency analysis
          const words = cleanedCell.split(/\s+/);
          words.forEach(word => {
            if (word) {
              const lowerWord = word.toLowerCase();
              wordFrequency[lowerWord] = (wordFrequency[lowerWord] || 0) + 1;
            }
          });
  
          // Collect data for each column
          if (!columnData[columns[cellIndex]]) {
            columnData[columns[cellIndex]] = [];
          }
          columnData[columns[cellIndex]].push(cleanedCell);
        });
      });
  
      const avgCellLength = totalCells ? (totalLength / totalCells).toFixed(2) : '0';
  
      // Determine the top 5 most common words
      const commonWords = Object.entries(wordFrequency)
        .sort((a, b) => b[1] - a[1])
        .slice(0, 5)
        .map(entry => entry[0]);
  
      // Generate column-specific summaries
      const columnSummaries = Object.entries(columnData).map(([column, data]) => {
        const uniqueValues = Array.from(new Set(data));
        const mostCommonValue = uniqueValues.sort((a, b) => data.filter(v => v === b).length - data.filter(v => v === a).length)[0];
        return `Column "${column}" has ${uniqueValues.length} unique values, most common value: "${mostCommonValue}"`;
      }).join('\n');
  
      const summary = nlp.generateComplexSentence(
        "The Excel file analysis reveals",
        `approximately ${rowCount} rows, ${columnCount} columns, an average cell length of ${avgCellLength} characters, and common words: ${commonWords.join(', ')}`,
        50
      );
  
      return `I've analyzed the Excel file. Here's a detailed overview:
  
  1. Number of rows: ${rowCount}
  2. Number of columns: ${columnCount}
  3. Total cells: ${totalCells}
  4. Average cell length: ${avgCellLength} characters
  5. Most common words: ${commonWords.join(', ')}
  
  Column-specific summaries:
  ${columnSummaries}
  
  ${summary}
  
  Please note that this analysis assumes a simple CSV structure. For more complex Excel files with multiple sheets or special formatting, a dedicated parsing library would be necessary.
  
  Would you like me to perform any specific analysis on this data?`;
    } catch (error) {
      console.error("Error processing Excel file:", error);
      return "I encountered an error while processing the Excel file. Please ensure it's a valid CSV-formatted Excel file.";
    }
  }
  
  // Utility function to parse a CSV line considering quoted commas
  function parseCSVLine(line: string): string[] {
    const result: string[] = [];
    let current = '';
    let inQuotes = false;
  
    for (let char of line) {
      if (char === '"' ) {
        inQuotes = !inQuotes;
      } else if (char === ',' && !inQuotes) {
        result.push(current);
        current = '';
      } else {
        current += char;
      }
    }
    result.push(current);
    return result;
  }
   
  async function processJsonFile(content: string): Promise<string> {
    try {
      const jsonData = JSON.parse(content);
      const keys = Object.keys(jsonData);
      const summary = nlp.generateComplexSentence("The JSON file contains", `${keys.length} top-level keys: ${keys.join(', ')}`, 50);
      return `I've analyzed the JSON file. ${summary}`;
    } catch (error) {
      return "I encountered an error while parsing the JSON file. Please make sure it's valid JSON.";
    }
  }
  
  interface NeuralNetworkLayer {
    neurons: number;
    activation: string;
  }
  
  interface NeuralNetworkStructure {
    inputLayer: NeuralNetworkLayer;
    hiddenLayers: NeuralNetworkLayer[];
    outputLayer: NeuralNetworkLayer;
  }
  
  export function getModelCalculations(input: string): string {
    const networkStructure: NeuralNetworkStructure = {
      inputLayer: { neurons: 64, activation: 'relu' },
      hiddenLayers: [
        { neurons: 128, activation: 'relu' },
        { neurons: 64, activation: 'relu' },
      ],
      outputLayer: { neurons: 32, activation: 'softmax' },
    };
  
    const tokenCount = input.split(/\s+/).length;
    const estimatedProcessingTime = (Math.random() * 0.5 + 0.1).toFixed(2);
  
    const calculations = `
  Analyzing input: "${input}"
  
  Linguistic Analysis:
  • Token count: ${tokenCount}
  • Estimated complexity: ${tokenCount < 10 ? 'Low' : tokenCount < 20 ? 'Moderate' : 'High'}
  
  Neural Network Architecture:
  • Input Layer: ${networkStructure.inputLayer.neurons} neurons (Activation: ${networkStructure.inputLayer.activation})
  ${networkStructure.hiddenLayers.map((layer, index) => 
    `• Hidden Layer ${index + 1}: ${layer.neurons} neurons (Activation: ${layer.activation})`
  ).join('\n')}
  • Output Layer: ${networkStructure.outputLayer.neurons} neurons (Activation: ${networkStructure.outputLayer.activation})
  
  Processing Pipeline:
  1. Semantic embedding of input tokens
  2. Forward propagation through neural layers
  3. Feature extraction and representation learning
  4. Output vector generation
  5. Response synthesis and natural language generation
  
  Computational Metrics:
  • Estimated processing time: ${estimatedProcessingTime} seconds
  • Theoretical floating-point operations: ~${(parseInt(estimatedProcessingTime) * 1e9).toExponential(2)}
  
  Note: This analysis provides a high-level overview of the model's structure and processing steps. Actual performance may vary based on hardware specifications and input complexity.
    `;
  
    return calculations.trim();
  }
  
  interface ChatHistory {
    id: string;
    name: string;
  }
  
  let chatHistories: ChatHistory[] = [];
  
  export function getChatHistories(): Promise<ChatHistory[]> {
    return new Promise((resolve) => {
      // Simulating an API call
      setTimeout(() => {
        resolve(chatHistories);
      }, 100);
    });
  }
        
  export function createChatHistory(name: string): Promise<void> {
    return new Promise((resolve) => {
      // Simulating an API call
      setTimeout(() => {
        const newHistory: ChatHistory = {
          id: Date.now().toString(),
          name,
        };
        chatHistories.push(newHistory);
        resolve();
      }, 100);
    });
  }
  
  export function renameChatHistory(id: string, newName: string): Promise<void> {
    return new Promise((resolve) => {
      // Simulating an API call
      setTimeout(() => {
        const history = chatHistories.find((h) => h.id === id);
        if (history) {
          history.name = newName;
        }
        resolve();
      }, 100);
    });
  }
  
  export function deleteChatHistory(id: string): Promise<void> {
    return new Promise((resolve) => {
      // Simulating an API call
      setTimeout(() => {
        chatHistories = chatHistories.filter((h) => h.id !== id);
        resolve();
      }, 100);
    });
  }
  
  // Add this function to your MazsAI.ts file
  interface Message {
    text: string;
    isUser: boolean;
    isTyping?: boolean;
    typingIndex?: number;
    attachments?: {
      file: File;
      url: string;
    }[];
    timestamp: Date;
    type?: 'text' | 'code'; // New field to specify message type
  }
  
  export function getChatHistoryMessages(id: string): Promise<Message[]> {
    return new Promise((resolve) => {
      // Simulating an API call to fetch messages for a specific chat history
      setTimeout(() => {
        // Replace this with actual logic to fetch messages
        const messages: Message[] = [
          { text: "Hello! I'm excited to start our conversation.", isUser: true, timestamp: new Date() },
          { text: "Hi there! It's great to meet you. I'm Mazs AI, your intelligent assistant. How can I assist you today? Feel free to ask me anything about AI, technology, or any other topic you're curious about.", isUser: false, timestamp: new Date() },
          { text: "That sounds interesting! Can you tell me more about your capabilities?", isUser: true, timestamp: new Date() },
          { text: "Certainly! I'm a versatile AI assistant capable of engaging in a wide range of tasks. I can help with information retrieval, answer questions, assist with problem-solving, provide explanations on complex topics, and even engage in creative writing. Is there a specific area you'd like to explore?", isUser: false, timestamp: new Date() }
        ];
        resolve(messages);
      }, 100);
    });
  }
  
