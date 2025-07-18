document.addEventListener("DOMContentLoaded", function () {
    let symptomData = document.getElementById("symptom-data");

    if (!symptomData) {
        console.error("❌ symptom-data element not found! Check Flask backend.");
        return;
    }

    let allSymptoms = JSON.parse(symptomData.textContent);
    console.log("✅ Symptoms Loaded in JavaScript:", allSymptoms);

    function filterSymptoms(inputId, dropdownId) {
        let input = document.getElementById(inputId);
        let dropdown = document.getElementById(dropdownId);
        let filter = input.value.toLowerCase().trim();
        dropdown.innerHTML = "";

        if (filter.length === 0) {
            dropdown.style.display = "none";
            return;
        }

        let matches = allSymptoms.filter(symptom => symptom.toLowerCase().includes(filter));

        if (matches.length === 0) {
            dropdown.style.display = "none";
            return;
        }

        matches.forEach(symptom => {
            let item = document.createElement("div");
            item.classList.add("dropdown-item");
            item.innerText = symptom;
            item.onclick = function () {
                input.value = symptom;
                dropdown.style.display = "none";
            };
            dropdown.appendChild(item);
        });

        dropdown.style.display = "block";
    }

    document.getElementById("symptom1").addEventListener("input", function() { filterSymptoms("symptom1", "dropdown1"); });
    document.getElementById("symptom2").addEventListener("input", function() { filterSymptoms("symptom2", "dropdown2"); });
    document.getElementById("symptom3").addEventListener("input", function() { filterSymptoms("symptom3", "dropdown3"); });
    document.getElementById("symptom4").addEventListener("input", function() { filterSymptoms("symptom4", "dropdown4"); });
});