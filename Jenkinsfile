
// Code Coverage JaCoCo working fine

def COLOR_MAP = [
    'SUCCESS': 'good',
    'FAILURE': 'danger',
]

pipeline {
    agent any
    tools {
        maven 'MAVEN3.9'
        jdk 'JDK17'
    }
    parameters {
        string(name: 'Checking_verbs', defaultValue: '4', description: 'Checking the critical verbs before build')
        booleanParam(name: 'RUN_DP_CHECK', defaultValue: true, description: 'Enable/Disable Dependency Check')
    }
    stages {
        stage('Cleanup Workspace') {
            steps {
                deleteDir()
            }
        }

        stage('Fetch Code') {
            steps {
                git branch: 'atom', url: 'https://github.com/hkhcoder/vprofile-project.git'
            }
        }

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

        stage('UNIT TEST & Code Coverage') {
            steps {
                sh 'mvn test jacoco:report'
            }
            post {
                success {
                    jacoco execPattern: 'target/jacoco.exec', classPattern: 'target/classes', sourcePattern: 'src/main/java', exclusionPattern: ''
                    archiveArtifacts artifacts: 'target/site/jacoco/index.html', fingerprint: true
                }
            }
        }

        stage('Checkstyle Analysis') {
            steps {
                sh 'mvn checkstyle:checkstyle'
            }
        }

        stage('Sonar Code Analysis') {
            environment {
                scannerHome = tool 'sonar6.2'
            }
            steps {
                withSonarQubeEnv('sonarserver') {
                    sh """${scannerHome}/bin/sonar-scanner \
                        -Dsonar.projectKey=vprofile \
                        -Dsonar.projectName=vprofile \
                        -Dsonar.projectVersion=1.0 \
                        -Dsonar.sources=src/ \
                        -Dsonar.java.binaries=target/classes \
                        -Dsonar.junit.reportsPath=target/surefire-reports/ \
                        -Dsonar.jacoco.reportsPath=target/jacoco.exec \
                        -Dsonar.java.checkstyle.reportPaths=target/checkstyle-result.xml"""
                }
            }
        }

        stage('Quality Gate') {
            steps {
                timeout(time: 1, unit: 'HOURS') {
                    waitForQualityGate abortPipeline: true
                }
            }
        }

        stage('Cyclomatic Complexity') {
            steps {
                sh 'lizard src/main/java > complexity_report.txt'
                archiveArtifacts 'complexity_report.txt'
            }
        }

        stage("Security Vulnerability Scan") {
            when {
                expression { return params.RUN_DP_CHECK }
            }
            steps {
                withCredentials([string(credentialsId: 'DP_API_KEY', variable: 'DP_API_KEY')]) {
                    script {
                        def jobName = env.JOB_NAME.replaceAll("[^a-zA-Z0-9]", "-")
                        def reportFile = "dependency-check-report-${jobName}.xml"

                        def args = [
                            "--format", "XML",
                            "--nvdApiKey", DP_API_KEY,
                            "--out", reportFile,
                            "--disableAssembly"
                        ]

                        dependencyCheck(
                            additionalArguments: args.join(" "),
                            odcInstallation: 'DP-Check'
                        )

                        dependencyCheckPublisher(
                            pattern: "**/${reportFile}",
                            failedTotalCritical: params.ALLOWED_CRITICAL_VUNERABILITIES
                        )
                    }
                }
            }
            post {
                always {
                    archiveArtifacts artifacts: "dependency-check-report-${env.JOB_NAME.replaceAll('[^a-zA-Z0-9]', '-')}.xml", fingerprint: true
                }
            }
        }

        stage('UploadArtifact') {
            steps {
                script {
                    def artifactExists = fileExists('target/vprofile-v2.war')
                    def reportExists = fileExists('complexity_report.txt')
                    def securityReportExists = fileExists("dependency-check-report-${env.JOB_NAME.replaceAll('[^a-zA-Z0-9]', '-')}.xml")
                    def coverageReportExists = fileExists('target/site/jacoco/index.html')

                    if (artifactExists && reportExists && securityReportExists && coverageReportExists) {
                        nexusArtifactUploader(
                            nexusVersion: 'nexus3',
                            protocol: 'http',
                            nexusUrl: '172.31.28.76:8081',
                            groupId: 'QA',
                            version: "${env.BUILD_ID}-${env.BUILD_TIMESTAMP}",
                            repository: 'vprofile-repo',
                            credentialsId: 'nexuslogin',
                            artifacts: [
                                [artifactId: 'vproapp', classifier: '', file: 'target/vprofile-v2.war', type: 'war'],
                                [artifactId: 'complexity-report', classifier: '', file: 'complexity_report.txt', type: 'txt'],
                                [artifactId: 'dependency-check-report', classifier: '', file: "dependency-check-report-${env.JOB_NAME.replaceAll('[^a-zA-Z0-9]', '-')}.xml", type: 'xml'],
                                [artifactId: 'code-coverage-report', classifier: '', file: 'target/site/jacoco/index.html', type: 'html']
                            ]
                        )
                    } else {
                        error("Required artifacts are missing. Skipping Nexus upload.")
                    }
                }
            }
        }
    }

    post {
        always {
            slackSend channel: 'jenkins-test-slack',
                color: COLOR_MAP[currentBuild.currentResult],
                message: "*${currentBuild.currentResult}:* Job ${env.JOB_NAME} build ${env.BUILD_NUMBER} \n Code Coverage Report: ${env.BUILD_URL}artifact/target/site/jacoco/index.html \n More info at: ${env.BUILD_URL}"
        }
    }
}
