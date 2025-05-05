<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Appointment Booked</title>
    <style>
        body {
            background: cyan;
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }

        .card {
            background: white;
            padding: 30px 50px;
            border-radius: 15px;
            box-shadow: 0 10px 25px rgba(0,0,0,0.1);
            animation: slideUp 0.8s ease-out;
            text-align: center;
        }

        @keyframes slideUp {
            from {
                opacity: 0;
                transform: translateY(50px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        h2 {
            color: #2c3e50;
        }

        .details {
            margin-top: 20px;
            font-size: 18px;
            color: #555;
        }

        .details span {
            display: block;
            margin: 10px 0;
        }
    </style>
</head>
<body>

<?php
if ($_SERVER["REQUEST_METHOD"] == "POST") {
    $name    = htmlspecialchars($_POST['name']);
    $email   = htmlspecialchars($_POST['email']);
    $phone   = htmlspecialchars($_POST['phone']);
    $date    = htmlspecialchars($_POST['date']);
    $slot    = htmlspecialchars($_POST['slot']);
    $problem = htmlspecialchars($_POST['problem']);

    echo '
    <div class="card">
        <h2>Your scheduled appointment has been booked successfully!</h2>
        <div class="details">
            <span><strong>Name:</strong> ' . $name . '</span>
            <span><strong>Email:</strong> ' . $email . '</span>
            <span><strong>Phone:</strong> ' . $phone . '</span>
            <span><strong>Date:</strong> ' . $date . '</span>
            <span><strong>Time Slot:</strong> ' . $slot . '</span>
            <span><strong>Problem:</strong> ' . $problem . '</span>
        </div>
    </div>';
} else {
    echo "<p>Invalid request.</p>";
}
?>

</body>
</html>
