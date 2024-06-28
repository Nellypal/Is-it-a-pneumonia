import React, { useState } from 'react';
import './App.css';

const Search: React.FC = () => {
    const [selectedFile, setSelectedFile] = useState<File | null>(null);
    const [selectedImage, setSelectedImage] = useState<string | ArrayBuffer | null>(null);
    const [isAnalyzing, setIsAnalyzing] = useState(false);
    const [pneumonia, setPneumonia] = useState<{ result: boolean} | null>(null)

    const handleImageChange = (event: React.ChangeEvent<HTMLInputElement>) => {
        if (event.target.files && event.target.files[0]) {
            const file = event.target.files[0];
            const reader = new FileReader();
            reader.onload = (e) => {
                if (e.target?.result) {
                    setSelectedImage(e.target.result);
                }
            };
            reader.readAsDataURL(file);
            setSelectedFile(file)
        }
    };

    const handleAnalyzeClick = () => {
        setIsAnalyzing(true);
        const getResult = async () => {
            if (!selectedFile)
                return
            setPneumonia(null);
            const formData = new FormData()
            formData.append("file", selectedFile)
            const result = await fetch("http://localhost:8000/is-pneumonia", {
                method: "POST",
                body: formData
            })
            if (result.status === 200) {
                const body = await result.json()
                setPneumonia(body)
            }
        }
        getResult()
        setTimeout(() => {
            // Simulate analysis complete after 3 seconds
            setIsAnalyzing(false);
            console.log('Analysis complete');
        }, 3000);
    };

    return (
        <div className="App">
            <header className="App-header">
                <h1 className="large-text">First, select an Image !</h1>
                <div className='image-wrapper'>
                    {selectedImage && <img src={selectedImage as string} alt="Selected" className="selected-image" />}
                    {isAnalyzing && <div className="progress-bar"></div>}
                </div>
                {!selectedImage && (
                    <label className="custom-button-search">
                        Browse
                        <input type="file" accept="image/*" onChange={handleImageChange} style={{ display: 'none' }} />
                    </label>
                )}
                <div>
                    {!pneumonia && selectedImage && !isAnalyzing && (
                        <button className="custom-button-analyze" onClick={handleAnalyzeClick}>
                            Analysons
                        </button>
                    )}
                </div>
                {!isAnalyzing && pneumonia && (<h2>
                    {pneumonia.result ? "PNEUMONIAAAA": "pas de pneumonia ^^"}
                </h2>)}
            </header>
        </div>
    );
}

export default Search;
