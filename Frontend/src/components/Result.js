import React from 'react';

export default function ResultCard({ ok, score, children }) {
  return (
    <section className={"result-card " + (ok ? "ok" : "bad")}>
      <div className="result-icon">{ok ? "✅" : "🛑"}</div>
      <h2 className="result-title">
        {ok ? "AI model didn't find anything malicious" : "AI model detected spam"}
      </h2>
      <div className="result-score">
        Confidence score: <strong>{Math.round(score * 100)}%</strong>
      </div>

      <div className="result-tabs">
        <div className="result-tab active">DETECTION</div>
        <div className="result-tab">DETAILS</div>
      </div>

      <div className="result-body">
        {children || (
          <p>
            Lorem ipsum dolor sit amet, consectetur adipiscing elit. Quisque maximus viverra augue,
            vel fringilla ipsum consequat eu. Donec egestas est a arcu tempus, vel venenatis quam
            laoreet.
          </p>
        )}
      </div>
    </section>
  );
}
