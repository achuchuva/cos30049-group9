import React, { useMemo, useState } from 'react';
import Tabs from '../components/Tabs';
import ResultCard from '../components/Result';

function fakePredict({ text, file, url }) {
  const base = text?.length || (url?.length ?? 32);
  const score = Math.min(0.98, Math.max(0.02, (base % 97) / 100));
  const ok = score < 0.5;
  return { ok, score };
}

export default function Detect() {
  const [tab, setTab] = useState('FILE');
  const [text, setText] = useState('');
  const [url, setUrl] = useState('');
  const [fileName, setFileName] = useState('');
  const [res, setRes] = useState(null);

  const canSubmit = useMemo(() => {
    if (tab === 'FILE') return Boolean(fileName);
    if (tab === 'TEXT') return text.trim().length > 0;
    if (tab === 'URL')  return url.trim().length > 0;
    return false;
  }, [tab, text, url, fileName]);

  const onSubmit = (e) => {
    e.preventDefault();
    const r = fakePredict({ text: tab==='TEXT'?text:'', url: tab==='URL'?url:'', file: tab==='FILE'?fileName:'' });
    setRes(r);
  };

  const onFile = (e) => {
    const f = e.target.files?.[0];
    if (f) setFileName(f.name);
  };

  return (
    <div>
      <section className="hero">
        <h1 className="page-title">Spam or Malware Detection</h1>
        <Tabs tabs={['FILE','TEXT','URL']} current={tab} onChange={setTab} />
      </section>

      <div className="container">
        <form onSubmit={onSubmit} className="detect-form">
          {tab === 'TEXT' && (
            <div className="field">
              <label>Spam Detection</label>
              <textarea placeholder="Enter potential spam" rows={5} value={text} onChange={e=>setText(e.target.value)} />
            </div>
          )}

          {tab === 'FILE' && (
            <div className="dropzone">
              <input id="file" type="file" onChange={onFile} />
              <label htmlFor="file">
                <div className="dz-in">
                  <p>Drag and drop your files<br/>here to upload</p>
                  <div className="or">OR</div>
                  <button type="button" className="btn">Browse Files</button>
                </div>
              </label>
              {fileName && <div className="upload-msg">File Uploaded: <strong>{fileName}</strong></div>}
            </div>
          )}

          {tab === 'URL' && (
            <div className="field">
              <label>URL</label>
              <input placeholder="https://example.com" value={url} onChange={e=>setUrl(e.target.value)} />
            </div>
          )}

          <button className="btn primary" disabled={!canSubmit}>Upload</button>
        </form>

        {res && <ResultCard ok={res.ok} score={res.score} />}
      </div>
    </div>
  );
}
