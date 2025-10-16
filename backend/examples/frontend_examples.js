"""
JavaScript/TypeScript examples for frontend integration.
"""

// Example 1: Single Prediction
async function predictSpam(text) {
  try {
    const response = await fetch('http://localhost:8000/api/v1/predict', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ text }),
    });
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    
    const data = await response.json();
    return data;
  } catch (error) {
    console.error('Error:', error);
    throw error;
  }
}

// Usage
predictSpam("WIN FREE CASH NOW!!!")
  .then(result => {
    console.log('Prediction:', result.prediction);
    console.log('Confidence:', result.confidence);
    console.log('Is Spam:', result.is_spam);
  })
  .catch(error => console.error('Error:', error));


// Example 2: Batch Prediction
async function predictBatch(texts) {
  try {
    const response = await fetch('http://localhost:8000/api/v1/predict/batch', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ texts }),
    });
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    
    const data = await response.json();
    return data;
  } catch (error) {
    console.error('Error:', error);
    throw error;
  }
}

// Usage
predictBatch([
  "Hello, how are you?",
  "WIN FREE CASH NOW!!!",
  "Meeting at 3pm"
])
  .then(result => {
    console.log('Total predictions:', result.total);
    result.predictions.forEach((pred, idx) => {
      console.log(`Text ${idx + 1}: ${pred.prediction} (${pred.confidence})`);
    });
  })
  .catch(error => console.error('Error:', error));


// Example 3: Health Check
async function checkHealth() {
  try {
    const response = await fetch('http://localhost:8000/health');
    const data = await response.json();
    return data;
  } catch (error) {
    console.error('Error:', error);
    throw error;
  }
}

// Usage
checkHealth()
  .then(status => {
    console.log('API Status:', status.status);
    console.log('Model Loaded:', status.model_loaded);
  });


// Example 4: React Hook for Spam Detection
import { useState } from 'react';

function useSpamDetection() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);

  const predict = async (text) => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await fetch('http://localhost:8000/api/v1/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text }),
      });
      
      if (!response.ok) {
        throw new Error('Prediction failed');
      }
      
      const data = await response.json();
      setResult(data);
      return data;
    } catch (err) {
      setError(err.message);
      throw err;
    } finally {
      setLoading(false);
    }
  };

  return { predict, loading, error, result };
}

// Usage in React component
function SpamDetector() {
  const [text, setText] = useState('');
  const { predict, loading, error, result } = useSpamDetection();

  const handleSubmit = async (e) => {
    e.preventDefault();
    await predict(text);
  };

  return (
    <div>
      <form onSubmit={handleSubmit}>
        <textarea 
          value={text} 
          onChange={(e) => setText(e.target.value)}
          placeholder="Enter text to check..."
        />
        <button type="submit" disabled={loading}>
          {loading ? 'Analyzing...' : 'Check for Spam'}
        </button>
      </form>
      
      {error && <div className="error">{error}</div>}
      
      {result && (
        <div className="result">
          <h3>Result: {result.prediction.toUpperCase()}</h3>
          <p>Confidence: {(result.confidence * 100).toFixed(2)}%</p>
          <p>Spam Probability: {(result.spam_probability * 100).toFixed(2)}%</p>
        </div>
      )}
    </div>
  );
}
