#!/bin/bash
concurrently "npm:server" "npm:api"