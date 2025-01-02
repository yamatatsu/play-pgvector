import * as cdk from 'aws-cdk-lib';
import * as ec2 from 'aws-cdk-lib/aws-ec2';
import * as rds from 'aws-cdk-lib/aws-rds';
import * as secretsmanager from 'aws-cdk-lib/aws-secretsmanager';
import * as iam from 'aws-cdk-lib/aws-iam';
import * as kms from 'aws-cdk-lib/aws-kms';
import { Duration } from 'aws-cdk-lib';

export class MyStack extends cdk.Stack {
  constructor(scope: cdk.Construct, id: string, props?: cdk.StackProps) {
    super(scope, id, props);

    const vpcCidr = '10.215.0.0/16';
    const dbPort = 5432;

    // VPC
    const vpc = new ec2.Vpc(this, 'VPC', {
      maxAzs: 2,
      natGateways: 0,
    });

    // Enhanced Monitoring Role
    const roleEnhancedMonitoring = new iam.Role(this, 'roleEnhancedMonitoring', {
      assumedBy: new iam.ServicePrincipal('monitoring.rds.amazonaws.com'),
      managedPolicies: [
        iam.ManagedPolicy.fromAwsManagedPolicyName('service-role/AmazonRDSEnhancedMonitoringRole'),
      ],
    });

    // KMS Key
    const encryptionKey = new kms.Key(this, 'EncryptionKey', {
      enableKeyRotation: true,
      alias: `alias/${this.stackName}`,
    });

    // DB Subnet Group
    const dbSubnetGroup = new rds.SubnetGroup(this, 'DBSubnetGroup', {
      vpc: vpc,
      description: 'RDS DB Subnet Group',
      subnets: [vpc.privateSubnets[0], vpc.privateSubnets[1]],
    });

    // DB Cluster Parameter Group
    const apgcustomclusterparamgroup = new rds.ClusterParameterGroup(this, 'apgcustomclusterparamgroup', {
      description: 'Aurora PostgreSQL Custom Cluster parameter group',
      family: 'aurora-postgresql15',
      parameters: {
        shared_preload_libraries: 'pg_stat_statements',
      },
    });

    // DB Parameter Group
    const apgcustomdbparamgroup = new rds.ParameterGroup(this, 'apgcustomdbparamgroup', {
      description: `${this.stackName}-dbparamgroup`,
      family: 'aurora-postgresql15',
      parameters: {
        log_rotation_age: '1440',
        log_rotation_size: '102400',
      },
    });

    // RDS Secrets
    const rdsSecrets = new secretsmanager.Secret(this, 'RDSSecrets', {
      secretName: 'apgpg-pgvector-secret',
      description: 'This is the secret for Aurora cluster',
      generateSecretString: {
        secretStringTemplate: JSON.stringify({ username: 'postgres' }),
        generateStringKey: 'password',
        passwordLength: 16,
        excludePunctuation: true,
      },
    });

    // Security Group
    const vpcSecurityGroup = new ec2.SecurityGroup(this, 'VPCSecurityGroup', {
      groupDescription: this.stackName,
      vpc: vpc,
      allowAllOutbound: true,
    });
    vpcSecurityGroup.addIngressRule(
      ec2.Peer.ipv4(vpcCidr),
      ec2.Port.tcp(dbPort),
      'Access to AppServer Host Security Group'
    );

    // DB Cluster
    const dbCluster = new rds.DatabaseCluster(this, 'DBCluster', {
      engine: rds.DatabaseClusterEngine.auroraPostgres({ version: rds.AuroraPostgresEngineVersion.VER_15_5 }),
      clusterIdentifier: 'apgpg-pgvector',
      port: dbPort,
      masterUser: {
        username: 'postgres',
        password: cdk.SecretValue.secretsManager(rdsSecrets.secretName, { jsonField: 'password' }),
      },
      defaultDatabaseName: 'apgpg',
      vpc: vpc,
      vpcSubnets: { subnets: vpc.privateSubnets },
      securityGroups: [vpcSecurityGroup],
      parameterGroup: apgcustomclusterparamgroup,
      instanceProps: {
        instanceType: ec2.InstanceType.of(ec2.InstanceClass.R6G, ec2.InstanceSize.XLARGE2),
        parameterGroup: apgcustomdbparamgroup,
      },
      storageEncrypted: true,
      kmsKey: encryptionKey,
      backup: {
        retention: Duration.days(7),
      },
      monitoringInterval: Duration.seconds(60),
      monitoringRole: roleEnhancedMonitoring,
      enablePerformanceInsights: true,
      performanceInsightRetention: rds.PerformanceInsightRetention.DEFAULT,
    });

    // DB Instances
    new rds.DatabaseInstance(this, 'DBNodeWriter', {
      cluster: dbCluster,
      instanceIdentifier: `${this.stackName}-node-01`,
      instanceType: ec2.InstanceType.of(ec2.InstanceClass.R6G, ec2.InstanceSize.XLARGE2),
      parameterGroup: apgcustomdbparamgroup,
      monitoringInterval: Duration.seconds(60),
      monitoringRole: roleEnhancedMonitoring,
      enablePerformanceInsights: true,
      performanceInsightRetention: rds.PerformanceInsightRetention.DEFAULT,
    });

    new rds.DatabaseInstance(this, 'DBNodeReader', {
      cluster: dbCluster,
      instanceIdentifier: `${this.stackName}-node-02`,
      instanceType: ec2.InstanceType.of(ec2.InstanceClass.R6G, ec2.InstanceSize.XLARGE2),
      parameterGroup: apgcustomdbparamgroup,
      monitoringInterval: Duration.seconds(60),
      monitoringRole: roleEnhancedMonitoring,
      enablePerformanceInsights: true,
      performanceInsightRetention: rds.PerformanceInsightRetention.DEFAULT,
    });

    // Outputs
    new cdk.CfnOutput(this, 'DBEndpoint', {
      description: 'Aurora PostgreSQL Endpoint',
      value: dbCluster.clusterEndpoint.hostname,
      exportName: `${this.stackName}-DBEndPoint`,
    });

    new cdk.CfnOutput(this, 'DBSecret', {
      description: 'Database Secret',
      value: rdsSecrets.secretArn,
      exportName: `${this.stackName}-DBSecrets`,
    });
  }
}

// The code below is needed to define the stack in your CDK application.
const app = new cdk.App();
new MyStack(app, 'MyStack');