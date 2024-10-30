import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import './LandingPage.css';
import logo from '../Assets/GMTStudio.png';
import mazsai from '../Assets/MazsAI.png';

const LandingPage: React.FC = () => {
  const [isScrolled, setIsScrolled] = useState(false);
  const [isMenuOpen, setIsMenuOpen] = useState(false);
     
  useEffect(() => {
    const handleScroll = () => {
      setIsScrolled(window.scrollY > 50);
    };
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  const toggleMenu = () => {
    setIsMenuOpen(!isMenuOpen);
  };

  return (
    <div className="min-h-screen bg-black flex flex-col">
      {/* Header */}
      <header className={`fixed w-full z-50 transition-all duration-500 backdrop-blur-lg ${isScrolled ? 'bg-black/80' : 'bg-transparent'}`}>
        <div className="max-w-7xl mx-auto px-6 py-4 flex justify-between items-center">
          <Link to="/" className="flex items-center space-x-2">
            <img src={logo} alt="Mazs AI Logo" className="h-8 w-auto rounded-full" />
            <h1 className="text-2xl font-medium text-white">GMTStudio</h1>
          </Link>
          <nav>
            <ul className="hidden md:flex space-x-8">
              <li><a href="#features" className="text-white/90 hover:text-white text-sm font-medium transition">Features</a></li>
              <li><a href="#about" className="text-white/90 hover:text-white text-sm font-medium transition">About</a></li>
              <li><a href="#contact" className="text-white/90 hover:text-white text-sm font-medium transition">Contact</a></li>
              <li>
                <Link to="/MazsAI" className="px-5 py-2 bg-white text-black rounded-full text-sm font-medium hover:bg-white/90 transition">
                  Try Now
                </Link>
              </li>
            </ul>
            <button 
              className="md:hidden text-white focus:outline-none" 
              onClick={toggleMenu} 
              aria-label="Toggle navigation menu"
              aria-expanded={isMenuOpen}
            >
              {isMenuOpen ? (
                <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              ) : (
                <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
                </svg>
              )}
            </button>
          </nav>
        </div>
        {/* Mobile Menu */}
        {isMenuOpen && (
          <div className="md:hidden bg-black/80 backdrop-blur-lg">
            <ul className="flex flex-col space-y-4 px-6 py-4">
              <li><a href="#features" className="text-white/90 hover:text-white text-sm font-medium transition">Features</a></li>
              <li><a href="#about" className="text-white/90 hover:text-white text-sm font-medium transition">About</a></li>
              <li><a href="#contact" className="text-white/90 hover:text-white text-sm font-medium transition">Contact</a></li>
              <li>
                <Link to="/MazsAI" className="block text-center px-5 py-2 bg-white text-black rounded-full text-sm font-medium hover:bg-white/90 transition">
                  Try Now
                </Link>
              </li>
            </ul>
          </div>
        )}
      </header>

      {/* Hero Section */}
      <section className="relative min-h-screen flex items-center justify-center bg-black pt-20 md:pt-0">
        <div className="absolute inset-0 bg-gradient-radial from-indigo-500/20 via-transparent to-transparent" />
        <div className="relative text-center text-white px-4 max-w-5xl mx-auto">
          <h2 className="text-4xl sm:text-6xl md:text-8xl font-bold mb-8 bg-clip-text text-transparent bg-gradient-to-r from-white to-indigo-300">
            Welcome to Mazs AI 
          </h2>
          <p className="text-lg sm:text-xl md:text-2xl mb-12 text-white/80 font-light max-w-3xl mx-auto leading-relaxed">
            Made by GMTStudio, for the future.
          </p>
          <div className="flex flex-col sm:flex-row justify-center gap-6">
            <Link to="/MazsAI" className="group px-6 sm:px-8 py-4 bg-white text-black text-lg font-medium rounded-full hover:bg-white/90 transition relative overflow-hidden">
              <span className="relative z-10">Get Started</span>
              <div className="absolute inset-0 bg-gradient-to-r from-indigo-500 to-purple-500 transform scale-x-0 group-hover:scale-x-100 transition-transform origin-left" />
            </Link>
            <a href="#features" className="px-6 sm:px-8 py-4 border border-white/30 text-white text-lg font-medium rounded-full hover:bg-white/10 transition backdrop-blur-sm">
              Learn More
            </a>
          </div>
        </div> 
        <div className="absolute bottom-10 left-1/2 transform -translate-x-1/2 animate-bounce">
          <svg className="w-6 h-6 text-white/50" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 14l-7 7m0 0l-7-7m7 7V3" />
          </svg>
        </div>
      </section>

      {/* Features Section */}
      <section id="features" className="py-16 md:py-32 bg-black">
        <div className="max-w-7xl mx-auto px-6">
          <div className="text-center mb-12 md:mb-20">
            <span className="text-indigo-400 font-medium text-sm uppercase tracking-wider">Features</span>
            <h3 className="text-3xl sm:text-5xl font-bold text-white mt-4">Intelligent Solutions</h3>
            <p className="text-md sm:text-xl text-white/70 mt-6 max-w-2xl mx-auto leading-relaxed">
              Discover what this AI can do for you.
            </p>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8 md:gap-12">
            {[
              {
                icon: (
                  <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                ),
                title: "Natural Language Processing",
                description: "This AI can understand a little bit of human language and generate human language."
              },
              {
                icon: (
                  <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                  </svg>
                ),
                title: "File Processing",
                description: "This AI can process and analyze files for Image all the way to Video."
              },
              {
                icon: (
                  <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                  </svg>
                ),
                title: "Smallest AI Model in the World",
                description: "This AI is the smallest AI model in the world, but not that good."
              }
            ].map((feature, index) => (
              <div key={index} className="group p-6 sm:p-8 bg-white/5 backdrop-blur-xl rounded-2xl hover:bg-white/10 transition-all duration-300">
                <div className="bg-gradient-to-br from-indigo-500 to-purple-500 rounded-xl p-4 w-16 h-16 mb-6 flex items-center justify-center">
                  {feature.icon}
                </div>
                <h4 className="text-lg sm:text-xl font-bold text-white mb-4">{feature.title}</h4>
                <p className="text-white/70 leading-relaxed">{feature.description}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* About Section */}
      <section id="about" className="py-16 md:py-32 bg-gradient-to-b from-black to-indigo-900">
        <div className="max-w-7xl mx-auto px-6">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-12 md:gap-16 items-center">
            <div className="space-y-8">
              <span className="text-indigo-400 font-medium text-sm uppercase tracking-wider">About Us</span>
              <h3 className="text-3xl sm:text-5xl font-bold text-white">Redefining Possibilities</h3>
              <p className="text-md sm:text-lg text-white/70 leading-relaxed">
                At Mazs AI, we're pushing the boundaries of what's possible with artificial intelligence. Our solutions are crafted to enhance human capabilities, not replace them, ensuring optimal performance and seamless integration across industries.
              </p>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-8">
                <div className="bg-white/5 backdrop-blur-xl p-6 rounded-2xl">
                  <h4 className="text-2xl sm:text-3xl font-bold text-indigo-400">About 1 Mb</h4>
                  <p className="text-white/70 mt-2">Super Lightweight</p>
                </div>
                <div className="bg-white/5 backdrop-blur-xl p-6 rounded-2xl">
                  <h4 className="text-2xl sm:text-3xl font-bold text-indigo-400">24/7</h4>
                  <p className="text-white/70 mt-2">Always Online</p>
                </div>
              </div>
            </div>
            <div className="relative flex justify-center">
              <img src={mazsai} alt="AI Technology" className="rounded-2xl shadow-2xl max-w-full h-auto" />
              <div className="absolute -bottom-8 -right-8 bg-gradient-to-r from-indigo-500 to-purple-500 text-white p-6 sm:p-8 rounded-2xl shadow-xl">
                <p className="text-lg sm:text-xl font-medium">Only 4610 Parameters</p>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Testimonials Section */}
      <section className="py-16 md:py-32 bg-black">
        <div className="max-w-7xl mx-auto px-6">
          <div className="text-center mb-12 md:mb-20">
            <span className="text-indigo-400 font-medium text-sm uppercase tracking-wider">Testimonials</span>
            <h3 className="text-3xl sm:text-5xl font-bold text-white mt-4">Success Stories</h3>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            {[
              {
                image: "/images/testimonial-1.jpg",
                name: "Alston Chang",
                role: "CEO of GMTStudio",
                quote: "Mazs AI has revolutionized our approach to data processing. The results have been extraordinary."
              },
              {
                image: "/images/testimonial-2.jpg",
                name: "Lucas Yeh",
                role: "Second CEO of GMTStudio",
                quote: "The seamless integration and powerful capabilities have transformed our development workflow."
              },
              {
                image: "/images/testimonial-3.jpg",
                name: "Willy Lin",
                role: "Third CEO of GMTStudio",
                quote: "Their AI solutions have streamlined our operations beyond our expectations."
              }
            ].map((testimonial, index) => (
              <div key={index} className="group p-6 sm:p-8 bg-white/5 backdrop-blur-xl rounded-2xl hover:bg-white/10 transition-all duration-300">
                <div className="flex items-center mb-4 sm:mb-6">
                  <img src={testimonial.image} alt={testimonial.name} className="w-12 h-12 sm:w-14 sm:h-14 rounded-full ring-2 ring-indigo-500 object-cover" />
                  <div className="ml-4">
                    <h5 className="text-md sm:text-lg font-medium text-white">{testimonial.name}</h5>
                    <p className="text-white/70">{testimonial.role}</p>
                  </div>
                </div>
                <p className="text-white/70 leading-relaxed mb-4 sm:mb-6">"{testimonial.quote}"</p>
                <div className="flex justify-center text-indigo-400">
                  {[...Array(5)].map((_, i) => (
                    <svg key={i} className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
                      <path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-.38-1.81.588-1.81h3.461a1 1 0 00.951-.69l1.07-3.292z" />
                    </svg>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>    
      </section>

      {/* Footer */}
      <footer className="bg-black/90 backdrop-blur-xl text-white/70 py-12 md:py-16 border-t border-white/10">
        <div className="max-w-7xl mx-auto px-6">
          <div className="grid grid-cols-1 md:grid-cols-4 gap-8 md:gap-12">
            <div>
              <Link to="/" className="flex items-center space-x-2 mb-6">
                <img src={logo} alt="Mazs AI Logo" className="h-8 w-auto rounded-full" />
                <span className="text-white text-xl font-medium">Mazs AI</span>
              </Link>
              <p>Empowering innovation through intelligent solutions.</p>
            </div>
            <div>
              <h4 className="text-white text-lg font-medium mb-4">Quick Links</h4>
              <ul className="space-y-3">
                <li><a href="#features" className="hover:text-white transition">Features</a></li>
                <li><a href="#about" className="hover:text-white transition">About</a></li>
                <li><a href="#contact" className="hover:text-white transition">Contact</a></li>
              </ul>
            </div>
            <div>
              <h4 className="text-white text-lg font-medium mb-4">Legal</h4>
              <ul className="space-y-3">
                <li><a href="#" className="hover:text-white transition">Privacy Policy</a></li>
                <li><a href="#" className="hover:text-white transition">Terms of Service</a></li>
                <li><a href="#" className="hover:text-white transition">Cookie Policy</a></li>
              </ul>
            </div>
            <div>
              <h4 className="text-white text-lg font-medium mb-4">Connect</h4>
              <div className="flex space-x-4">
                {[
                  {
                    icon: <path d="M24 12.073c0-6.627-5.373-12-12-12s-12 5.373-12 12c0 5.99 4.388 10.954 10.125 11.854v-8.385H7.078v-3.47h3.047V9.43c0-3.007 1.792-4.669 4.533-4.669 1.312 0 2.686.235 2.686.235v2.953H15.83c-1.491 0-1.956.925-1.956 1.874v2.25h3.328l-.532 3.47h-2.796v8.385C19.612 23.027 24 18.062 24 12.073z" />,
                    href: "#"
                  },
                  {
                    icon: <path d="M23.953 4.57a10 10 0 01-2.825.775 4.958 4.958 0 002.163-2.723c-.951.555-2.005.959-3.127 1.184a4.92 4.92 0 00-8.384 4.482C7.69 8.095 4.067 6.13 1.64 3.162a4.822 4.822 0 00-.666 2.475c0 1.71.87 3.213 2.188 4.096a4.904 4.904 0 01-2.228-.616v.06a4.923 4.923 0 003.946 4.827 4.996 4.996 0 01-2.212.085 4.936 4.936 0 004.604 3.417 9.867 9.867 0 01-6.102 2.105c-.39 0-.779-.023-1.17-.067a13.995 13.995 0 007.557 2.209c9.053 0 13.998-7.496 13.998-13.985 0-.21 0-.42-.015-.63A9.935 9.935 0 0024 4.59z" />,
                    href: "#"
                  },
                  {
                    icon: <path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433c-1.144 0-2.063-.926-2.063-2.065 0-1.138.92-2.063 2.063-2.063 1.14 0 2.064.925 2.064 2.063 0 1.139-.925 2.065-2.064 2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0h.003z" />,
                    href: "#"
                  }
                ].map((social, index) => (
                  <a key={index} href={social.href} className="text-white/70 hover:text-white transition" aria-label={`Connect via ${social.href}`}>
                    <svg className="w-6 h-6" fill="currentColor" viewBox="0 0 24 24">
                      {social.icon}
                    </svg>
                  </a>
                ))}
              </div>
            </div>
          </div>
          <div className="border-t border-white/10 mt-8 md:mt-12 pt-6 md:pt-8 text-center">
            <p>&copy; {new Date().getFullYear()} Mazs AI. All rights reserved.</p>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default LandingPage;