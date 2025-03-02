# Jenkins CI/CD Pipeline for Code Quality, Coverage, Complexity & Security

## Objective
The goal of this assignment is to create a Jenkins pipeline that integrates various open-source tools to assess code coverage, code quality, cyclomatic complexity, and security vulnerabilities. This project demonstrates the ability to create an effective CI/CD process that aids in maintaining high-quality code standards in a software development project.

---

## Prerequisites for Setting Up the Pipeline
### Server Setup
1. **Provision an AWS EC2 Instance**:
   - Select an Ubuntu or Amazon Linux AMI.
   - Ensure necessary security group rules are configured (SSH, HTTP, and Jenkins-specific ports).
   
2. **Install Required Dependencies**:
   ```sh
   sudo apt update && sudo apt upgrade -y
   sudo apt install -y openjdk-17-jdk maven git
   ```

3. **Install Jenkins**:
   ```sh
   wget -q -O - https://pkg.jenkins.io/debian-stable/jenkins.io.key | sudo tee /usr/share/keyrings/jenkins-keyring.asc > /dev/null
   echo "deb [signed-by=/usr/share/keyrings/jenkins-keyring.asc] https://pkg.jenkins.io/debian-stable binary/" | sudo tee /etc/apt/sources.list.d/jenkins.list > /dev/null
   sudo apt update
   sudo apt install -y jenkins
   sudo systemctl start jenkins
   sudo systemctl enable jenkins
   ```

4. **Configure Jenkins**:
   - Access Jenkins UI: `http://<JENKINS_SERVER_IP>:8080`
   - Retrieve the admin password:
     ```sh
     sudo cat /var/lib/jenkins/secrets/initialAdminPassword
     ```
   - Install recommended plugins and create an admin user.
   - Configure necessary Jenkins plugins for Git, SonarQube, and Nexus integration.

---

## Setting Up the Pipeline
### Source Code Management
- Use Git as the SCM tool.
- Configure Jenkins to pull the source code from the GitHub repository:
  ```groovy
  pipeline {
      agent any
      stages {
          stage('Fetch Code') {
              steps {
                  git branch: 'main', url: 'https://github.com/hkhcoder/vprofile-project.git'
              }
          }
      }
  }
  ```

### Webhook Configuration (GitHub → Jenkins)
To trigger the pipeline automatically on code changes:
1. **Enable Webhook in GitHub**:
   - Go to the GitHub repository.
   - Navigate to **Settings → Webhooks → Add Webhook**.
   - Enter the **Jenkins Webhook URL**: `http://<JENKINS_SERVER_IP>:8080/github-webhook/`
   - Set **Content type** to `application/json`.
   - Choose **Just the push event**.
   - Click **Add Webhook**.

2. **Configure Jenkins to Respond to Webhooks**:
   - Install **GitHub Integration Plugin**.
   - In **Jenkins → Manage Jenkins → Configure System**, ensure **GitHub Webhook Trigger** is enabled.
   - In your Jenkins job, select **Build Triggers → GitHub hook trigger for GITScm polling**.

---

## Pipeline Stages
### 1. Cleanup Workspace
```groovy
stage('Cleanup Workspace') {
    steps {
        deleteDir()
    }
}
```

### 2. Build the Application
```groovy
stage('Build') {
    steps {
        sh 'mvn clean install -DskipTests'
    }
    post {
        success {
            archiveArtifacts artifacts: '**/target/*.war'
        }
    }
}
```

### 3. Code Quality Analysis with SonarQube
```groovy
stage('Sonar Code Analysis') {
    environment {
        scannerHome = tool 'sonar6.2'
    }
    steps {
        withSonarQubeEnv('sonarserver') {
            sh """${scannerHome}/bin/sonar-scanner \
                -Dsonar.projectKey=vprofile \
                -Dsonar.sources=src/ \
                -Dsonar.java.binaries=target/classes"""
        }
    }
}
```

### 4. Code Coverage with JaCoCo
```groovy
stage('UNIT TEST & Code Coverage') {
    steps {
        sh 'mvn test jacoco:report'
    }
    post {
        success {
            jacoco execPattern: 'target/jacoco.exec', classPattern: 'target/classes', sourcePattern: 'src/main/java'
        }
    }
}
```

### 5. Cyclomatic Complexity Check
```groovy
stage('Cyclomatic Complexity') {
    steps {
        sh 'lizard src/main/java > complexity_report.txt'
        archiveArtifacts 'complexity_report.txt'
    }
}
```

### 6. Security Vulnerability Scan
```groovy
stage('Security Vulnerability Scan') {
    steps {
        sh 'dependency-check --scan src/main/java --out reports/'
    }
}
```

### 7. Upload Artifacts to Nexus Repository
```groovy
stage('UploadArtifact') {
    steps {
        script {
            nexusArtifactUploader(
                nexusVersion: 'nexus3',
                protocol: 'http',
                nexusUrl: '172.31.28.76:8081',
                groupId: 'QA',
                version: "${env.BUILD_ID}-${env.BUILD_TIMESTAMP}",
                repository: 'vprofile-repo',
                credentialsId: 'nexuslogin',
                artifacts: [[artifactId: 'vproapp', file: 'target/vprofile-v2.war', type: 'war']]
            )
        }
    }
}
```
![jenkins-final-output.png]

### 8. Notifications
```groovy
post {
    always {
        slackSend channel: 'jenkins-test-slack',
            color: COLOR_MAP[currentBuild.currentResult],
            message: "*${currentBuild.currentResult}:* Job ${env.JOB_NAME} build ${env.BUILD_NUMBER}"
    }
}
```
![slack-notification-jenkins-trigger.png]

---

## Troubleshooting
- **Jenkins Webhook Not Triggering**: Ensure GitHub webhook is correctly configured and Jenkins has the correct webhook URL.
- **SonarQube Analysis Failing**: Check SonarQube server connectivity and authentication.
- **Maven Build Errors**: Ensure all dependencies are properly configured in `pom.xml`.
- **Security Scan Timeout**: Increase timeout or check network connectivity.

---

## Conclusion
This pipeline ensures high code quality by integrating code coverage, quality checks, complexity analysis, and security scanning. The CI/CD process automates the testing and deployment lifecycle, making it efficient and reliable.

