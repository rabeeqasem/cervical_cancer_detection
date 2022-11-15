import React, { useState } from 'react';
import { NavLink, useLocation } from 'react-router-dom';
import { IoIosPeople } from 'react-icons/io';
import { GiBrain } from 'react-icons/gi';

const menuItems = [
  {
    path: "/",
    name: "Introduction",
    icon: <IoIosPeople/>
  },
  {
    path: "/prediction",
    name: "Prediction",
    icon: <GiBrain/>
  }
];

const Sidebar = ({children}) => {
  const [isOpen, setIsOpen] = useState(false);
  const location = useLocation();

  return (
    <div className="flex">
      <div
        className={` ${isOpen ? "w-64" : "w-20 "} bg-teal-500 h-screen p-5 pt-8 fixed duration-300 z-50`}
      >
        <img
          src={require("../images/control.png")}
          alt="ControlArrow"
          className={`absolute cursor-pointer -right-4 top-9 w-7 border-teal-500 border-2 rounded-full  ${!isOpen && "rotate-180"}`}
          onClick={() => setIsOpen(!isOpen)}
        />
        <div className="flex gap-x-4 items-center">
          <div className={`bg-white rounded-lg py-5 ${isOpen ? 'w-32 bg-center' : 'w-20 hover:bg-right'} duration-300 logo-bg`}>
          </div>
        </div>
        <div className="pt-6">
          {
            menuItems.map((item, index) => (
              <NavLink 
                to={item.path} 
                key={index} 
                className={`flex rounded-lg p-2 my-2 cursor-pointer hover:bg-red-500 text-white text-sm items-center gap-x-4 ${location.pathname === item.path && 'active-class'} transition-all duration-300`}
                onClick={() => setIsOpen(true)}
              >
                <div className={`text-2xl ${!isOpen && 'mx-auto'}`}>{item.icon}</div>
                <div className={`text-xl ${isOpen ? 'block' : 'invisible'}`}>{item.name}</div>
              </NavLink>
            ))
          }
        </div>
      </div>
      <div className="w-full p-8 ml-20" onClick={() => setIsOpen(false)}>{children}</div>
    </div>
  )
}

export default Sidebar