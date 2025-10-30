import React from 'react';

export default function Tabs({ tabs, current, onChange }) {
  return (
    <div className="tabs">
      {tabs.map(t => (
        <button
          key={t}
          className={"tab" + (current === t ? " active" : "")}
          onClick={() => onChange(t)}
          type="button"
        >
          {t}
        </button>
      ))}
    </div>
  );
}
