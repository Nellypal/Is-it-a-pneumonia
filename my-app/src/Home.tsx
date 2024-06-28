import React from 'react';
import { useNavigate } from 'react-router-dom';
import './App.css';

const Home: React.FC = () => {
    const navigate = useNavigate();

    const handleClick = () => {
        navigate('/search');
    }

    return (
        <header className="App-header home">
            <div>
                <h1 className="small-text">&lt; Do I have a &gt;</h1>
                <h1 className="large-text">pneumonia ?</h1>
            </div>
            <button className="custom-button" onClick={handleClick}>Let's check !</button>
        </header>
    );
}

export default Home;
