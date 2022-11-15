import React from 'react';
import {BrowserRouter, Route, Routes} from 'react-router-dom';
import Sidebar from '../../components/Sidebar';
import Introduction from './Introduction';
import Prediction from './Prediction';
import "./dashboard.css";

const Dashboard = () => {
  return (
    <div>
      <BrowserRouter>
        <Sidebar>
          <Routes>
            <Route path="/" element={<Introduction title="Introduction"/>}/>
            <Route path="/prediction" element={<Prediction title="Prediction"/>}/>
          </Routes>
        </Sidebar>
      </BrowserRouter>
    </div>
  )
}

export default Dashboard