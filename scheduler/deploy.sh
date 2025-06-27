#!/bin/bash

# RSE Scheduler Orchestrator Deployment Script
# Automated deployment for development, staging, and production environments

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_NAME="rse-scheduler"
SERVICE_USER="rse-scheduler"
INSTALL_DIR="/opt/rse-scheduler"
LOG_DIR="/var/log/rse-scheduler"
DATA_DIR="/var/lib/rse-scheduler"
BACKUP_DIR="/var/backups/rse-scheduler"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Help function
show_help() {
    cat << EOF
RSE Scheduler Orchestrator Deployment Script

Usage: $0 [OPTIONS] COMMAND

Commands:
    install         Install the RSE Scheduler Orchestrator
    update          Update existing installation
    start           Start the service
    stop            Stop the service
    restart         Restart the service
    status          Show service status
    logs            Show service logs
    backup          Create backup
    restore         Restore from backup
    uninstall       Remove the installation
    health          Check system health

Options:
    -e, --env ENV           Environment (development|staging|production) [default: production]
    -u, --user USER         Service user [default: rse-scheduler]
    -d, --dir DIR           Installation directory [default: /opt/rse-scheduler]
    -p, --port PORT         Service port [default: 9090]
    -m, --method METHOD     Deployment method (systemd|pm2|docker) [default: systemd]
    -c, --config FILE       Custom configuration file
    -b, --backup-dir DIR    Backup directory [default: /var/backups/rse-scheduler]
    -f, --force             Force operation without confirmation
    -v, --verbose           Verbose output
    -h, --help              Show this help message

Examples:
    $0 install --env production
    $0 update --force
    $0 start
    $0 logs --follow
    $0 backup

EOF
}

# Parse command line arguments
ENVIRONMENT="production"
DEPLOY_METHOD="systemd"
PORT="9090"
FORCE=false
VERBOSE=false
CONFIG_FILE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -e|--env)
            ENVIRONMENT="$2"
            shift 2
            ;;
        -u|--user)
            SERVICE_USER="$2"
            shift 2
            ;;
        -d|--dir)
            INSTALL_DIR="$2"
            shift 2
            ;;
        -p|--port)
            PORT="$2"
            shift 2
            ;;
        -m|--method)
            DEPLOY_METHOD="$2"
            shift 2
            ;;
        -c|--config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        -b|--backup-dir)
            BACKUP_DIR="$2"
            shift 2
            ;;
        -f|--force)
            FORCE=true
            shift
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        -*)
            log_error "Unknown option: $1"
            show_help
            exit 1
            ;;
        *)
            COMMAND="$1"
            shift
            ;;
    esac
done

# Verbose mode
if [[ "$VERBOSE" == "true" ]]; then
    set -x
fi

# Check if running as root
check_root() {
    if [[ $EUID -ne 0 ]]; then
        log_error "This script must be run as root"
        exit 1
    fi
}

# Check system requirements
check_requirements() {
    log_info "Checking system requirements..."
    
    # Check OS
    if [[ ! -f /etc/os-release ]]; then
        log_error "Unsupported operating system"
        exit 1
    fi
    
    # Check Node.js
    if ! command -v node &> /dev/null; then
        log_error "Node.js is not installed"
        exit 1
    fi
    
    # Check npm
    if ! command -v npm &> /dev/null; then
        log_error "npm is not installed"
        exit 1
    fi
    
    # Check systemd (if using systemd)
    if [[ "$DEPLOY_METHOD" == "systemd" ]] && ! command -v systemctl &> /dev/null; then
        log_error "systemd is not available"
        exit 1
    fi
    
    # Check PM2 (if using PM2)
    if [[ "$DEPLOY_METHOD" == "pm2" ]] && ! command -v pm2 &> /dev/null; then
        log_warning "PM2 is not installed. Installing..."
        npm install -g pm2
    fi
    
    # Check Docker (if using Docker)
    if [[ "$DEPLOY_METHOD" == "docker" ]] && ! command -v docker &> /dev/null; then
        log_error "Docker is not installed"
        exit 1
    fi
    
    log_success "System requirements check passed"
}

# Create service user
create_user() {
    if ! id "$SERVICE_USER" &>/dev/null; then
        log_info "Creating service user: $SERVICE_USER"
        useradd --system --shell /bin/false --home "$INSTALL_DIR" --create-home "$SERVICE_USER"
        log_success "Service user created"
    else
        log_info "Service user already exists: $SERVICE_USER"
    fi
}

# Create directories
create_directories() {
    log_info "Creating directories..."
    
    mkdir -p "$INSTALL_DIR"
    mkdir -p "$LOG_DIR"
    mkdir -p "$DATA_DIR"
    mkdir -p "$BACKUP_DIR"
    
    # Set ownership
    chown -R "$SERVICE_USER:$SERVICE_USER" "$INSTALL_DIR"
    chown -R "$SERVICE_USER:$SERVICE_USER" "$LOG_DIR"
    chown -R "$SERVICE_USER:$SERVICE_USER" "$DATA_DIR"
    chown -R "$SERVICE_USER:$SERVICE_USER" "$BACKUP_DIR"
    
    # Set permissions
    chmod 755 "$INSTALL_DIR"
    chmod 755 "$LOG_DIR"
    chmod 755 "$DATA_DIR"
    chmod 755 "$BACKUP_DIR"
    
    log_success "Directories created"
}

# Install application
install_application() {
    log_info "Installing RSE Scheduler Orchestrator..."
    
    # Copy files
    cp -r "$SCRIPT_DIR"/* "$INSTALL_DIR/"
    
    # Install dependencies
    cd "$INSTALL_DIR"
    npm ci --only=production
    
    # Set up configuration
    if [[ -n "$CONFIG_FILE" ]]; then
        cp "$CONFIG_FILE" "$INSTALL_DIR/.env"
    elif [[ -f "$SCRIPT_DIR/.env.example" ]]; then
        cp "$SCRIPT_DIR/.env.example" "$INSTALL_DIR/.env"
        log_warning "Please configure $INSTALL_DIR/.env before starting the service"
    fi
    
    # Set ownership
    chown -R "$SERVICE_USER:$SERVICE_USER" "$INSTALL_DIR"
    
    log_success "Application installed"
}

# Install systemd service
install_systemd_service() {
    log_info "Installing systemd service..."
    
    # Copy service file
    cp "$SCRIPT_DIR/rse-scheduler.service" /etc/systemd/system/
    
    # Update service file with actual paths
    sed -i "s|/opt/rse-scheduler|$INSTALL_DIR|g" /etc/systemd/system/rse-scheduler.service
    sed -i "s|User=rse-scheduler|User=$SERVICE_USER|g" /etc/systemd/system/rse-scheduler.service
    sed -i "s|Group=rse-scheduler|Group=$SERVICE_USER|g" /etc/systemd/system/rse-scheduler.service
    
    # Reload systemd
    systemctl daemon-reload
    systemctl enable rse-scheduler.service
    
    log_success "Systemd service installed"
}

# Install PM2 service
install_pm2_service() {
    log_info "Installing PM2 service..."
    
    # Start with PM2
    sudo -u "$SERVICE_USER" pm2 start "$INSTALL_DIR/ecosystem.config.js" --env "$ENVIRONMENT"
    
    # Save PM2 configuration
    sudo -u "$SERVICE_USER" pm2 save
    
    # Install PM2 startup script
    pm2 startup systemd -u "$SERVICE_USER" --hp "$INSTALL_DIR"
    
    log_success "PM2 service installed"
}

# Install Docker service
install_docker_service() {
    log_info "Installing Docker service..."
    
    cd "$INSTALL_DIR"
    
    # Build Docker image
    docker build -t "$PROJECT_NAME:latest" .
    
    # Start with Docker Compose
    docker-compose up -d
    
    log_success "Docker service installed"
}

# Main install function
install() {
    check_root
    check_requirements
    create_user
    create_directories
    install_application
    
    case "$DEPLOY_METHOD" in
        systemd)
            install_systemd_service
            ;;
        pm2)
            install_pm2_service
            ;;
        docker)
            install_docker_service
            ;;
        *)
            log_error "Unknown deployment method: $DEPLOY_METHOD"
            exit 1
            ;;
    esac
    
    log_success "RSE Scheduler Orchestrator installed successfully!"
    log_info "Configuration file: $INSTALL_DIR/.env"
    log_info "Log directory: $LOG_DIR"
    log_info "Data directory: $DATA_DIR"
    log_info "Backup directory: $BACKUP_DIR"
}

# Update function
update() {
    log_info "Updating RSE Scheduler Orchestrator..."
    
    # Stop service
    stop_service
    
    # Backup current installation
    backup
    
    # Update application
    install_application
    
    # Start service
    start_service
    
    log_success "Update completed"
}

# Service control functions
start_service() {
    case "$DEPLOY_METHOD" in
        systemd)
            systemctl start rse-scheduler.service
            ;;
        pm2)
            sudo -u "$SERVICE_USER" pm2 start "$PROJECT_NAME"
            ;;
        docker)
            cd "$INSTALL_DIR" && docker-compose start
            ;;
    esac
    log_success "Service started"
}

stop_service() {
    case "$DEPLOY_METHOD" in
        systemd)
            systemctl stop rse-scheduler.service
            ;;
        pm2)
            sudo -u "$SERVICE_USER" pm2 stop "$PROJECT_NAME"
            ;;
        docker)
            cd "$INSTALL_DIR" && docker-compose stop
            ;;
    esac
    log_success "Service stopped"
}

restart_service() {
    case "$DEPLOY_METHOD" in
        systemd)
            systemctl restart rse-scheduler.service
            ;;
        pm2)
            sudo -u "$SERVICE_USER" pm2 restart "$PROJECT_NAME"
            ;;
        docker)
            cd "$INSTALL_DIR" && docker-compose restart
            ;;
    esac
    log_success "Service restarted"
}

status_service() {
    case "$DEPLOY_METHOD" in
        systemd)
            systemctl status rse-scheduler.service
            ;;
        pm2)
            sudo -u "$SERVICE_USER" pm2 status "$PROJECT_NAME"
            ;;
        docker)
            cd "$INSTALL_DIR" && docker-compose ps
            ;;
    esac
}

# Logging function
show_logs() {
    case "$DEPLOY_METHOD" in
        systemd)
            journalctl -u rse-scheduler.service -f
            ;;
        pm2)
            sudo -u "$SERVICE_USER" pm2 logs "$PROJECT_NAME"
            ;;
        docker)
            cd "$INSTALL_DIR" && docker-compose logs -f
            ;;
    esac
}

# Backup function
backup() {
    log_info "Creating backup..."
    
    BACKUP_FILE="$BACKUP_DIR/rse-scheduler-$(date +%Y%m%d-%H%M%S).tar.gz"
    
    tar -czf "$BACKUP_FILE" -C "$INSTALL_DIR" .
    
    log_success "Backup created: $BACKUP_FILE"
}

# Restore function
restore() {
    if [[ $# -eq 0 ]]; then
        log_error "Please specify backup file to restore"
        exit 1
    fi
    
    BACKUP_FILE="$1"
    
    if [[ ! -f "$BACKUP_FILE" ]]; then
        log_error "Backup file not found: $BACKUP_FILE"
        exit 1
    fi
    
    log_info "Restoring from backup: $BACKUP_FILE"
    
    # Stop service
    stop_service
    
    # Extract backup
    tar -xzf "$BACKUP_FILE" -C "$INSTALL_DIR"
    
    # Set ownership
    chown -R "$SERVICE_USER:$SERVICE_USER" "$INSTALL_DIR"
    
    # Start service
    start_service
    
    log_success "Restore completed"
}

# Health check function
health_check() {
    log_info "Performing health check..."
    
    # Check service status
    if ! status_service &>/dev/null; then
        log_error "Service is not running"
        return 1
    fi
    
    # Check HTTP endpoint
    if command -v curl &>/dev/null; then
        if curl -f "http://localhost:$PORT/health" &>/dev/null; then
            log_success "Health check endpoint is responding"
        else
            log_error "Health check endpoint is not responding"
            return 1
        fi
    fi
    
    # Check log files
    if [[ -f "$LOG_DIR/rse-scheduler.log" ]]; then
        log_success "Log files are present"
    else
        log_warning "Log files not found"
    fi
    
    log_success "Health check completed"
}

# Uninstall function
uninstall() {
    if [[ "$FORCE" != "true" ]]; then
        read -p "Are you sure you want to uninstall RSE Scheduler Orchestrator? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log_info "Uninstall cancelled"
            exit 0
        fi
    fi
    
    log_info "Uninstalling RSE Scheduler Orchestrator..."
    
    # Stop and disable service
    case "$DEPLOY_METHOD" in
        systemd)
            systemctl stop rse-scheduler.service || true
            systemctl disable rse-scheduler.service || true
            rm -f /etc/systemd/system/rse-scheduler.service
            systemctl daemon-reload
            ;;
        pm2)
            sudo -u "$SERVICE_USER" pm2 delete "$PROJECT_NAME" || true
            ;;
        docker)
            cd "$INSTALL_DIR" && docker-compose down || true
            docker rmi "$PROJECT_NAME:latest" || true
            ;;
    esac
    
    # Remove directories
    rm -rf "$INSTALL_DIR"
    rm -rf "$LOG_DIR"
    rm -rf "$DATA_DIR"
    
    # Remove user
    userdel "$SERVICE_USER" || true
    
    log_success "Uninstall completed"
}

# Main command dispatcher
case "${COMMAND:-}" in
    install)
        install
        ;;
    update)
        update
        ;;
    start)
        start_service
        ;;
    stop)
        stop_service
        ;;
    restart)
        restart_service
        ;;
    status)
        status_service
        ;;
    logs)
        show_logs
        ;;
    backup)
        backup
        ;;
    restore)
        restore "$@"
        ;;
    health)
        health_check
        ;;
    uninstall)
        uninstall
        ;;
    "")
        log_error "No command specified"
        show_help
        exit 1
        ;;
    *)
        log_error "Unknown command: $COMMAND"
        show_help
        exit 1
        ;;
esac