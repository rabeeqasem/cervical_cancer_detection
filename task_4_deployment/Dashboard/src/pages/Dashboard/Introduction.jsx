import React, { useEffect } from 'react';
import { NavLink } from 'react-router-dom';
import { BiRightArrowAlt } from 'react-icons/bi';

const Introduction = ({title}) => {
  useEffect(() => {
    document.title = "DrCADx | " + title || "";
  }, [title]);

  return (
    <div className="grid grid-rows-3">
      <div className="flex justify-between">
        <img 
          src={require("../../images/Dr-CADx.png")} 
          alt="DrCADx" 
          className="w-48 h-16"
        />
        <img 
          src={require("../../images/omdena.png")} 
          alt="Omdena"
          className="w-60 h-16"
        />
      </div>
      <div className="flex flex-col justify-center">
        <div className="mx-auto mb-2 bg-teal-500 p-3 text-white text-8xl rounded-2xl outline outline-2 outline-red-400 hover:outline-red-500 outline-offset-2 hover:outline-offset-8 transition-all duration-300">CervAI</div>
        <div className="mx-auto text-2xl text-gray-500">An app for reporting Cervical Cytology using Bethesda System</div>
        <NavLink 
          to="/prediction"
          className="mx-auto mt-3 text-lg flex items-center hover:text-red-500"
        >
          <div className="hover:mr-3 transition-all duration-300">Click here to proceed</div> 
          <BiRightArrowAlt size={30} className="transition-all duration-500"/>
        </NavLink>
      </div>
      <div className="flex">
        <div className="self-end text-gray-400">Disclaimer: This document is strictly private, confidential and personal to its recipients and should not be copied, distributed or reproduced in whole or in part, nor passed to any third party.</div>
      </div>
    </div>
  )
}

export default Introduction;