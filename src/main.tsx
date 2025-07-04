// import { StrictMode } from 'react'
// import { createRoot } from 'react-dom/client'
// import './index.css'
// // import App from './App.tsx'
// import BibliometricDashboard from './BibliometricDashboard'

// createRoot(document.getElementById('root')!).render(
//   <StrictMode>
//     {/* <App /> */}
//     <BibliometricDashboard />
//   </StrictMode>,
// )

// =============

import React from 'react';
import ReactDOM from 'react-dom/client';
import BibliometricDashboard from './BibliometricDashboard';
import './index.css';

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <BibliometricDashboard />
  </React.StrictMode>
);